"""Messages a user is owed, and how far each of their devices has read.

An ephemeral channel reaches whoever is connected at that instant, which is
right for a cursor blinking or someone typing — nobody wants yesterday's
keystrokes on reconnect. It is wrong for anything a user is owed: a phone in a
tunnel misses it entirely and nothing records that it did.

A durable channel writes first and fans out second, keeps one ordered inbox
per user, and a separate cursor per device. One inbox and many cursors is what
makes multiple devices work: a laptop reading a message does not consume it
from the phone, and a device that was offline asks for what came after the
last sequence it saw.

The inbox is shared rather than per replica, because a device reconnects
wherever the load balancer sends it, which is rarely the process that held it
before.
"""

import asyncio

from base.models.definitions.realtime import (
    INBOX_LOCK_TTL_SECONDS,
    INBOX_RETENTION_SECONDS,
    REALTIME_CURSOR_NAMESPACE,
    REALTIME_INBOX_LOCK_NAMESPACE,
    REALTIME_INBOX_NAMESPACE,
)
from base.models.view_models.realtime import RealtimeEnvelope
from base.services.cache import AbstractCacheService
from base.services.cache.factory import get_cache_service
from base.tools.tenancy import TenantContext

LOCK_ATTEMPTS = 50
LOCK_WAIT_SECONDS = 0.02


class DurableInbox:
    """One ordered inbox per user, and one cursor per device.

    Not a singleton: it is a collaborator the connection manager owns, and it
    must use the manager's cache. A shared instance would bind to whichever
    cache happened to be built first, which in a test run is rarely the one
    the caller passed.
    """

    def __init__(self, cache: AbstractCacheService | None = None) -> None:
        self._cache = cache

    @property
    def cache(self) -> AbstractCacheService:
        """Resolved on first use: settings are not registered at construction."""
        if self._cache is None:
            self._cache = get_cache_service()
        return self._cache

    @staticmethod
    def inbox_key(user_id: str) -> str:
        """Scoped by tenant where there is one, so two never share an inbox.

        The user id alone would already be unique. The tenant is in the key
        anyway, because an inbox is the kind of thing that gets listed and
        migrated, and a key that cannot name the wrong customer is cheaper
        than remembering to filter every time.
        """
        tenant = TenantContext.get()
        return f"{tenant}:{user_id}" if tenant else user_id

    async def aappend(
        self, user_id: str, envelope: RealtimeEnvelope
    ) -> RealtimeEnvelope:
        """Record a message and give it its place in the order.

        The sequence is assigned under a lock because two replicas can publish
        to the same user at the same moment, and two messages sharing a number
        would make a cursor ambiguous: a device resuming from it would either
        miss one or receive one twice, and nothing would say which.
        """
        key = self.inbox_key(user_id)
        await self._aacquire(key)
        try:
            existing = await self.cache.aget_list_raw(
                key, namespace=REALTIME_INBOX_NAMESPACE
            )
            numbered = envelope.model_copy(update={"seq": len(existing) + 1})
            await self.cache.aappend_to_list(
                key,
                namespace=REALTIME_INBOX_NAMESPACE,
                value=numbered.model_dump(mode="json", by_alias=True),
                ttl_seconds=INBOX_RETENTION_SECONDS,
            )
            return numbered
        finally:
            await self.cache.arelease_lock(key, namespace=REALTIME_INBOX_LOCK_NAMESPACE)

    async def asince(self, user_id: str, after_seq: int) -> list[RealtimeEnvelope]:
        """Everything the caller has not seen, oldest first.

        Strictly after: a device sends the last sequence it has, not the first
        it wants, because the former is the only number it can be sure about.

        A device claiming a sequence the inbox does not reach has outlived it
        — retention expired, or it was cleared — and numbering has restarted
        under it. Everything held is new to that device, and filtering on a
        number from the previous epoch would hand back nothing until the count
        climbed past it, which is silence exactly where it looks like health.
        """
        stored = await self.cache.aget_list_raw(
            self.inbox_key(user_id), namespace=REALTIME_INBOX_NAMESPACE
        )
        envelopes = [RealtimeEnvelope.model_validate_json(raw) for raw in stored]
        if after_seq > len(envelopes):
            return envelopes
        return [e for e in envelopes if e.seq is not None and e.seq > after_seq]

    async def alatest_seq(self, user_id: str) -> int:
        """The highest sequence this user's inbox holds. Zero when empty."""
        stored = await self.cache.aget_list_raw(
            self.inbox_key(user_id), namespace=REALTIME_INBOX_NAMESPACE
        )
        return len(stored)

    async def acursor(self, user_id: str, device_id: str) -> int:
        """How far one device has read. Zero when it has never acknowledged."""
        stored = await self.cache.aget(
            self._cursor_key(user_id, device_id),
            namespace=REALTIME_CURSOR_NAMESPACE,
        )
        return int(stored) if stored else 0

    async def aadvance(self, user_id: str, device_id: str, seq: int) -> int:
        """Move one device's cursor forward, never back.

        An acknowledgement that arrives out of order, or twice, must not undo
        progress: the cursor is the high-water mark, not the last thing seen.
        """
        current = await self.acursor(user_id, device_id)
        if seq <= current:
            return current
        await self.cache.aset(
            self._cursor_key(user_id, device_id),
            namespace=REALTIME_CURSOR_NAMESPACE,
            value=str(seq),
            ttl_seconds=INBOX_RETENTION_SECONDS,
        )
        return seq

    async def aclear(self, user_id: str) -> None:
        """Drop a user's inbox. Cursors expire on their own."""
        await self.cache.aclear_list(
            self.inbox_key(user_id), namespace=REALTIME_INBOX_NAMESPACE
        )

    def _cursor_key(self, user_id: str, device_id: str) -> str:
        return f"{self.inbox_key(user_id)}:{device_id}"

    async def _aacquire(self, key: str) -> None:
        """Wait for the lock, and go ahead without it rather than hang.

        A lock that cannot be taken means another replica is mid-append or one
        died holding it. Blocking forever would stall the publisher; the worst
        case of proceeding is a repeated sequence, which is better than a
        socket that never gets anything again.
        """
        for _ in range(LOCK_ATTEMPTS):
            if await self.cache.aacquire_lock(
                key,
                namespace=REALTIME_INBOX_LOCK_NAMESPACE,
                ttl_seconds=INBOX_LOCK_TTL_SECONDS,
            ):
                return
            await asyncio.sleep(LOCK_WAIT_SECONDS)


__all__ = ["DurableInbox"]
