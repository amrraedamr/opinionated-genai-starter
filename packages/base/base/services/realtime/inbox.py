"""Durable delivery: one ordered inbox per user, one read cursor per device."""

from base.models.definitions.realtime import (
    INBOX_RETENTION_SECONDS,
    REALTIME_CURSOR_NAMESPACE,
)
from base.models.view_models.realtime import InboxEntry, RealtimeEnvelope
from base.services.cache import AbstractCacheService
from base.services.cache.factory import get_cache_service
from base.services.store.realtime_inbox import RealtimeInboxStore
from base.services.store.realtime_inbox.cache import CacheRealtimeInboxStore
from base.tools.tenancy import TenantContext


class DurableInbox:
    """Records durable messages and tracks how far each device has read."""

    def __init__(
        self,
        cache: AbstractCacheService | None = None,
        store: RealtimeInboxStore | None = None,
    ) -> None:
        self._cache = cache
        self._store = store or CacheRealtimeInboxStore(cache)

    @property
    def cache(self) -> AbstractCacheService:
        if self._cache is None:
            self._cache = get_cache_service()
        return self._cache

    @staticmethod
    def inbox_key(user_id: str) -> str:
        """Scoped by tenant where there is one."""
        tenant = TenantContext.get()
        return f"{tenant}:{user_id}" if tenant else user_id

    async def aappend(
        self, user_id: str, envelope: RealtimeEnvelope
    ) -> RealtimeEnvelope:
        """Record a message and return it carrying its sequence."""
        key = self.inbox_key(user_id)
        entry = InboxEntry(
            sequence=await self._store.anext_sequence(key), envelope=envelope
        )
        await self._store.aappend(key, [entry])
        return entry.delivered()

    async def asince(self, user_id: str, after_seq: int) -> list[RealtimeEnvelope]:
        """Everything after `after_seq`, oldest first; all of it if that is stale."""
        key = self.inbox_key(user_id)
        entries = await self._store.aafter(key, after_seq)
        if not entries and after_seq > 0:
            held = await self._store.aget(key, limit=1)
            if not held or held[-1].sequence < after_seq:
                entries = await self._store.aget(key)
        return [e.delivered() for e in entries]

    async def acursor(self, user_id: str, device_id: str) -> int:
        """How far one device has read. Zero when it has never acknowledged."""
        stored = await self.cache.aget(
            self._cursor_key(user_id, device_id),
            namespace=REALTIME_CURSOR_NAMESPACE,
        )
        return int(stored) if stored else 0

    async def aadvance(self, user_id: str, device_id: str, seq: int) -> int:
        """Move one device's cursor forward, never back."""
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
        await self._store.aclear(self.inbox_key(user_id))

    def _cursor_key(self, user_id: str, device_id: str) -> str:
        return f"{self.inbox_key(user_id)}:{device_id}"


__all__ = ["DurableInbox"]
