"""Per-replica socket registry, with pub/sub fan-out that reaches every replica."""

import asyncio
import contextlib
import time
from collections import defaultdict
from datetime import UTC, datetime

from base.models.definitions.realtime import (
    PRESENCE_REFRESH_SECONDS,
    PRESENCE_RETENTION_SECONDS,
    PRESENCE_TTL_SECONDS,
    REALTIME_FANOUT_CHANNEL,
    REALTIME_LAST_SEEN_NAMESPACE,
    REALTIME_PRESENCE_NAMESPACE,
    ChannelDelivery,
    RealtimeCloseCode,
    RealtimeMessageType,
)
from base.models.view_models.exceptions import CacheError
from base.models.view_models.realtime import (
    FanoutMessage,
    Presence,
    RealtimeEnvelope,
)
from base.services.cache import AbstractCacheService
from base.services.cache.factory import get_cache_service
from base.services.common.meta import Singleton
from base.services.logging.console import ConsoleLoggerService
from base.services.realtime.connection import RealtimeConnection
from base.services.realtime.inbox import DurableInbox


class ConnectionManager(metaclass=Singleton):
    """The sockets this replica holds, and the fan-out that reaches the rest."""

    def __init__(self, cache: AbstractCacheService | None = None) -> None:
        self._cache = cache
        self._logger = ConsoleLoggerService()
        self._by_user: dict[str, dict[str, RealtimeConnection]] = defaultdict(dict)
        self._listener: asyncio.Task[None] | None = None
        self._refresher: asyncio.Task[None] | None = None
        self._draining = False
        self._inbox = DurableInbox(cache)

    @property
    def cache(self) -> AbstractCacheService:
        """Resolved on first use: settings are not registered at construction."""
        if self._cache is None:
            self._cache = get_cache_service()
        return self._cache

    @property
    def draining(self) -> bool:
        return self._draining

    def devices_of(self, user_id: str) -> list[str]:
        """Which of a user's clients this replica is holding."""
        return list(self._by_user.get(user_id, {}))

    def local_count(self) -> int:
        return sum(len(devices) for devices in self._by_user.values())

    async def astart(self) -> None:
        """Begin listening for messages published by any replica, including us."""
        if self._listener is not None:
            return
        self._draining = False
        self._listener = asyncio.create_task(self._alisten())
        self._refresher = asyncio.create_task(self._arefresh())

    async def aregister(self, connection: RealtimeConnection) -> bool:
        """Take ownership of a socket. False when this replica is going away."""
        if self._draining:
            return False
        user = self._user_key(connection)
        existing = self._by_user[user].get(connection.device_id)
        if existing is not None and existing is not connection:
            await existing.aclose(
                RealtimeCloseCode.GOING_AWAY, "Replaced by a newer connection"
            )
        self._by_user[user][connection.device_id] = connection
        await self._amark(user, [connection])
        return True

    async def aunregister(self, connection: RealtimeConnection) -> None:
        """Forget a socket, without disturbing a replacement that took its place."""
        user = self._user_key(connection)
        devices = self._by_user.get(user, {})
        if devices.get(connection.device_id) is connection:
            devices.pop(connection.device_id)
        if not devices:
            self._by_user.pop(user, None)
        await self._aforget(user, connection)

    async def apublish_to_user(
        self,
        user_id: str,
        envelope: RealtimeEnvelope,
        delivery: ChannelDelivery = ChannelDelivery.EPHEMERAL,
    ) -> RealtimeEnvelope:
        """Send to every device this user has, wherever it is connected; durable ones are recorded first."""
        if delivery == ChannelDelivery.DURABLE:
            envelope = await self._inbox.aappend(user_id, envelope)
        await self.cache.apublish(
            REALTIME_FANOUT_CHANNEL,
            FanoutMessage(user_id=user_id, envelope=envelope).jsonify(),
        )
        return envelope

    async def aresume(self, connection: RealtimeConnection, last_seq: int = 0) -> int:
        """Send a reconnecting device whatever it missed. Returns how many."""
        user = self._user_key(connection)
        missed = await self._inbox.asince(user, last_seq)
        for envelope in missed:
            await connection.asend(envelope)
        return len(missed)

    async def aacknowledge(self, connection: RealtimeConnection, seq: int) -> int:
        """Record that one device has read up to `seq`. Returns its cursor."""
        return await self._inbox.aadvance(
            self._user_key(connection), connection.device_id, seq
        )

    async def acursor_of(self, connection: RealtimeConnection) -> int:
        """How far this device has acknowledged."""
        return await self._inbox.acursor(
            self._user_key(connection), connection.device_id
        )

    async def ahandle(
        self, connection: RealtimeConnection, envelope: RealtimeEnvelope
    ) -> RealtimeEnvelope | None:
        """Settle the delivery frames here; hand application frames back."""
        match envelope.type:
            case RealtimeMessageType.ACK:
                if envelope.seq is not None:
                    await self.aacknowledge(connection, envelope.seq)
                return None
            case RealtimeMessageType.RESUME:
                await self.aresume(connection, envelope.seq or 0)
                return None
            case _:
                return envelope

    async def adeliver_locally(self, user_id: str, envelope: RealtimeEnvelope) -> int:
        """Write to the open sockets this replica holds. Returns how many took it."""
        delivered = 0
        for connection in list(self._by_user.get(user_id, {}).values()):
            if connection.closed:
                continue
            await connection.asend(envelope)
            delivered += not connection.closed
        return delivered

    async def apresence(self, user_id: str) -> Presence:
        """Whether a user has a live socket on any replica, and when they were last seen."""
        sockets = await self.cache.aget_hashset_all(
            user_id, namespace=REALTIME_PRESENCE_NAMESPACE
        )
        cutoff = time.time() - PRESENCE_TTL_SECONDS
        live = [float(at) for at in (sockets or {}).values() if float(at) > cutoff]
        if live:
            return Presence(
                user_id=user_id,
                online=True,
                last_seen=datetime.fromtimestamp(max(live), UTC),
            )
        left = await self.cache.aget(user_id, namespace=REALTIME_LAST_SEEN_NAMESPACE)
        return Presence(
            user_id=user_id,
            online=False,
            last_seen=datetime.fromisoformat(left) if left else None,
        )

    async def adrain(self, reason: str = "Server is shutting down") -> None:
        """Stop taking sockets, tell the ones we hold to go elsewhere, close."""
        self._draining = True
        closing = RealtimeEnvelope(
            type=RealtimeMessageType.CLOSING, payload={"reason": reason}
        )
        for user, devices in list(self._by_user.items()):
            for connection in list(devices.values()):
                await connection.asend(closing)
                await connection.aclose(RealtimeCloseCode.GOING_AWAY, reason)
                await self._aforget(user, connection)
        self._by_user.clear()
        await self.astop()

    async def astop(self) -> None:
        """Stop listening and refreshing. Safe to call when it never started."""
        for task in (self._listener, self._refresher):
            if task is None:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._listener = self._refresher = None

    async def _alisten(self) -> None:
        """Deliver whatever any replica published, this one included."""
        async for raw in self.cache.subscribe(REALTIME_FANOUT_CHANNEL):
            try:
                message = FanoutMessage.model_validate_json(raw)
            except ValueError:
                self._logger.warning("Realtime fan-out carried an unreadable message")
                continue
            await self.adeliver_locally(message.user_id, message.envelope)

    async def _arefresh(self) -> None:
        """Keep this replica's open sockets counted for as long as they stay open."""
        while True:
            await asyncio.sleep(PRESENCE_REFRESH_SECONDS)
            for user, devices in list(self._by_user.items()):
                open_ = [c for c in devices.values() if not c.closed]
                try:
                    await self._amark(user, open_)
                except CacheError as exc:
                    self._logger.warning("Realtime presence refresh failed: %s", exc)

    async def _amark(self, user_id: str, connections: list[RealtimeConnection]) -> None:
        """Record these sockets as live now, one hash field each."""
        if not connections:
            return
        now = time.time()
        await self.cache.aset_hashset(
            user_id,
            namespace=REALTIME_PRESENCE_NAMESPACE,
            hashset={str(c.connection_id): now for c in connections},
            ttl_seconds=PRESENCE_TTL_SECONDS,
        )

    async def _aforget(self, user_id: str, connection: RealtimeConnection) -> None:
        """Drop one socket's field, leaving any other replica's, and note when it left."""
        await self.cache.adelete_hashset_pairs(
            user_id, REALTIME_PRESENCE_NAMESPACE, str(connection.connection_id)
        )
        await self.cache.aset(
            user_id,
            namespace=REALTIME_LAST_SEEN_NAMESPACE,
            value=datetime.now(UTC).isoformat(),
            ttl_seconds=PRESENCE_RETENTION_SECONDS,
        )

    @staticmethod
    def _user_key(connection: RealtimeConnection) -> str:
        """Who a socket belongs to, falling back to the token's subject."""
        return str(connection.user_id) if connection.user_id else connection.subject


__all__ = ["ConnectionManager"]
