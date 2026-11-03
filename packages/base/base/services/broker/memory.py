import asyncio
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from base.services.broker import AbstractBroker, Handler
from base.services.common.meta import SingletonABC
from base.services.logging.console import ConsoleLoggerService
from contracts.events import Event


@dataclass
class _Delivery:
    event: Event
    attempts: int = 0
    group: str | None = None


class InMemoryBroker(AbstractBroker, metaclass=SingletonABC):
    """Development and local-stack transport. Not for production use."""

    SEEN_LIMIT = 10_000

    def __init__(self, *, max_attempts: int = 3, retry_backoff: float = 0.01) -> None:
        self._handlers: dict[str, list[tuple[str, Handler]]] = defaultdict(list)
        self._queue: asyncio.Queue[_Delivery] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._dlq: dict[str, list[Event]] = defaultdict(list)
        self._max_attempts = max_attempts
        self._retry_backoff = retry_backoff
        self._stopping = asyncio.Event()
        self._logger = ConsoleLoggerService()

    def _mark_seen(self, key: str) -> None:
        """Bounded, so a long-running process cannot grow this without limit."""
        self._seen[key] = None
        while len(self._seen) > self.SEEN_LIMIT:
            self._seen.popitem(last=False)

    async def _apublish(self, event: Event) -> None:
        await self._queue.put(_Delivery(event))

    async def _asubscribe(self, topic: str, group: str, handler: Handler) -> None:
        self._handlers[topic].append((group, handler))

    async def _astart(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._consume())

    async def _astop(self) -> None:
        self._stopping.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=5)
        except TimeoutError:
            self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _consume(self) -> None:
        while not self._stopping.is_set() or not self._queue.empty():
            try:
                delivery = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            event = delivery.event
            handlers = [
                (group, handler)
                for group, handler in self._handlers.get(event.topic, [])
                if delivery.group is None or delivery.group == group
            ]
            for group, handler in handlers:
                key = f"{group}:{event.meta.event_id}"
                if key in self._seen:
                    continue
                try:
                    await handler(event)
                    self._mark_seen(key)
                except Exception:
                    attempts = delivery.attempts + 1
                    if attempts >= self._max_attempts:
                        self._dlq[event.topic].append(event)
                        self._logger.exception(
                            "Handler exhausted retries for %s in group %s",
                            event.topic,
                            group,
                        )
                    else:
                        await asyncio.sleep(self._retry_backoff * 2 ** (attempts - 1))
                        await self._queue.put(_Delivery(event, attempts, group))
            self._queue.task_done()

    async def _areplay_dlq(self, topic: str, *, limit: int = 100) -> int:
        events = self._dlq[topic][:limit]
        del self._dlq[topic][:limit]
        for event in events:
            await self._queue.put(_Delivery(event))
        return len(events)

    async def adrain(self) -> None:
        await self._queue.join()

    async def _ahealth_check(self) -> bool:
        """In-process, so reachable whenever the object exists."""
        return True

    def dlq_size(self, topic: str) -> int:
        return len(self._dlq[topic])
