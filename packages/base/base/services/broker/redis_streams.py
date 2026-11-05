import asyncio
import json
from datetime import UTC, datetime

from base.services.broker import AbstractBroker, Handler
from base.services.common.meta import SingletonABC
from base.services.configuration import get_settings, settings_for
from base.services.configuration.protocols import (
    SupportsRedisBrokerConfig,
)
from base.services.logging.console import ConsoleLoggerService
from base.tools.errors import MissingExtraError
from contracts.events import Event

try:
    from redis.asyncio import Redis
    from redis.exceptions import ResponseError
except ModuleNotFoundError as exc:
    raise MissingExtraError("redis", "redis", feature="RedisStreamsBroker") from exc


class RedisStreamsBroker(AbstractBroker, metaclass=SingletonABC):
    def __init__(self, config: SupportsRedisBrokerConfig | None = None) -> None:
        self._config: SupportsRedisBrokerConfig = config or settings_for(SupportsRedisBrokerConfig)
        self._redis: Redis | None = None
        self._subscriptions: list[tuple[str, str, Handler]] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._logger = ConsoleLoggerService()
        self._consumer = f"{get_settings().service_name}-{id(self)}"
        self._stopping = asyncio.Event()
        self._max_attempts = self._config.broker_max_attempts
        self._claim_idle_ms = self._config.broker_claim_idle_ms
        self._retry_backoff = self._config.broker_retry_backoff_seconds

    @property
    def client(self) -> Redis:
        if self._redis is None:
            self._redis = Redis.from_url(
                self._config.redis_url,
                decode_responses=True,
            )
        return self._redis

    async def _apublish(self, event: Event) -> None:
        await self.client.xadd(event.topic, {"event": event.model_dump_json()})

    async def _asubscribe(self, topic: str, group: str, handler: Handler) -> None:
        self._subscriptions.append((topic, group, handler))

    async def _astart(self) -> None:
        if self._tasks:
            return
        self._stopping.clear()
        for topic, group, handler in self._subscriptions:
            try:
                await self.client.xgroup_create(topic, group, id="0", mkstream=True)
            except ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise
            self._tasks.append(asyncio.create_task(self._consume(topic, group, handler)))

    async def _astop(self) -> None:
        self._stopping.set()
        if self._tasks:
            done, pending = await asyncio.wait(self._tasks, timeout=5)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
        self._tasks.clear()
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def _consume(self, topic: str, group: str, handler: Handler) -> None:
        """A transport error must not kill the consumer: a dead task would stop
        consuming this topic silently and forever."""
        while not self._stopping.is_set():
            try:
                await self._reclaim(topic, group, handler)
                entries = await self.client.xreadgroup(
                    group, self._consumer, {topic: ">"}, count=16, block=1000
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception(
                    "Broker read failed for %s in group %s; retrying", topic, group
                )
                await self._abackoff()
                continue
            for _stream, messages in entries or []:
                for message_id, fields in messages:
                    await self._deliver(topic, group, message_id, fields, handler)

    async def _abackoff(self) -> None:
        """Wait before reconnecting, but wake immediately on shutdown."""
        try:
            await asyncio.wait_for(self._stopping.wait(), self._retry_backoff)
        except TimeoutError:
            pass

    async def _reclaim(self, topic: str, group: str, handler: Handler) -> None:
        _next_id, messages, _deleted = await self.client.xautoclaim(
            topic,
            group,
            self._consumer,
            min_idle_time=self._claim_idle_ms,
            start_id="0-0",
            count=16,
        )
        for message_id, fields in messages:
            await self._deliver(topic, group, message_id, fields, handler)

    async def _deliver(
        self,
        topic: str,
        group: str,
        message_id: str,
        fields: dict[str, str],
        handler: Handler,
    ) -> None:
        try:
            event = Event.model_validate(json.loads(fields["event"]))
            await handler(event)
            await self.client.xack(topic, group, message_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            pending = await self.client.xpending_range(topic, group, message_id, message_id, 1)
            deliveries = int(pending[0].get("times_delivered", 1)) if pending else 1
            if deliveries >= self._max_attempts:
                await self.client.xadd(
                    f"{topic}.dlq",
                    {
                        "event": fields.get("event", ""),
                        "source_topic": topic,
                        "source_message_id": message_id,
                        "consumer_group": group,
                        "attempts": str(deliveries),
                        "failed_at": datetime.now(UTC).isoformat(),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    },
                )
                await self.client.xack(topic, group, message_id)
                return
            await asyncio.sleep(min(self._retry_backoff * 2 ** (deliveries - 1), 30.0))
            self._logger.exception("Handler failed for %s in group %s", topic, group)

    async def _areplay_dlq(self, topic: str, *, limit: int = 100) -> int:
        dlq = f"{topic}.dlq"
        entries = await self.client.xrange(dlq, min="-", max="+", count=limit)
        replayed = 0
        for message_id, fields in entries:
            raw = fields.get("event")
            if not raw:
                continue
            event = Event.model_validate(json.loads(raw))
            await self._apublish(event)
            await self.client.xdel(dlq, message_id)
            replayed += 1
        return replayed

    async def _ahealth_check(self) -> bool:
        try:
            return bool(await self.client.ping())
        except Exception:
            self._logger.exception("Broker health check failed")
            return False
