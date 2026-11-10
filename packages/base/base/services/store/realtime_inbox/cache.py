from base.models.definitions.realtime import (
    INBOX_MAX_ENTRIES,
    INBOX_RETENTION_SECONDS,
    REALTIME_INBOX_NAMESPACE,
    REALTIME_INBOX_SEQUENCE_NAMESPACE,
)
from base.models.view_models.realtime import InboxEntry
from base.services.cache import AbstractCacheService
from base.services.cache.factory import get_cache_service
from base.services.store.realtime_inbox import RealtimeInboxStore


class CacheRealtimeInboxStore(RealtimeInboxStore):
    """Inbox in the shared cache, so any replica can resume any device."""

    def __init__(self, cache: AbstractCacheService | None = None) -> None:
        self._cache = cache

    @property
    def cache(self) -> AbstractCacheService:
        if self._cache is None:
            self._cache = get_cache_service()
        return self._cache

    # Append

    def _append(self, parent_id: str, items: list[InboxEntry]) -> list[InboxEntry]:
        for item in items:
            self.cache.add_scored(
                parent_id,
                namespace=REALTIME_INBOX_NAMESPACE,
                value=item.model_dump(mode="json"),
                score=item.sequence,
                ttl_seconds=INBOX_RETENTION_SECONDS,
                max_length=INBOX_MAX_ENTRIES,
            )
        return items

    async def _aappend(
        self, parent_id: str, items: list[InboxEntry]
    ) -> list[InboxEntry]:
        for item in items:
            await self.cache.aadd_scored(
                parent_id,
                namespace=REALTIME_INBOX_NAMESPACE,
                value=item.model_dump(mode="json"),
                score=item.sequence,
                ttl_seconds=INBOX_RETENTION_SECONDS,
                max_length=INBOX_MAX_ENTRIES,
            )
        return items

    # Get

    def _get(self, parent_id: str, *, limit: int | None = None) -> list[InboxEntry]:
        raw = self.cache.get_scored(parent_id, namespace=REALTIME_INBOX_NAMESPACE)
        return self._tail([InboxEntry.model_validate(v) for v in raw], limit)

    async def _aget(
        self, parent_id: str, *, limit: int | None = None
    ) -> list[InboxEntry]:
        raw = await self.cache.aget_scored(
            parent_id, namespace=REALTIME_INBOX_NAMESPACE
        )
        return self._tail([InboxEntry.model_validate(v) for v in raw], limit)

    def _after(self, parent_id: str, sequence: int) -> list[InboxEntry]:
        raw = self.cache.get_scored(
            parent_id, namespace=REALTIME_INBOX_NAMESPACE, above=sequence
        )
        return [InboxEntry.model_validate(v) for v in raw]

    async def _aafter(self, parent_id: str, sequence: int) -> list[InboxEntry]:
        raw = await self.cache.aget_scored(
            parent_id, namespace=REALTIME_INBOX_NAMESPACE, above=sequence
        )
        return [InboxEntry.model_validate(v) for v in raw]

    @staticmethod
    def _tail(items: list[InboxEntry], limit: int | None) -> list[InboxEntry]:
        if limit is None or limit >= len(items):
            return items
        return items[-limit:] if limit > 0 else []

    # Count

    def _count(self, parent_id: str) -> int:
        return len(self.cache.get_scored(parent_id, namespace=REALTIME_INBOX_NAMESPACE))

    async def _acount(self, parent_id: str) -> int:
        return len(
            await self.cache.aget_scored(parent_id, namespace=REALTIME_INBOX_NAMESPACE)
        )

    # Next sequence

    def _next_sequence(self, parent_id: str) -> int:
        return self.cache.increment(
            parent_id,
            namespace=REALTIME_INBOX_SEQUENCE_NAMESPACE,
            ttl_seconds=INBOX_RETENTION_SECONDS,
        )

    async def _anext_sequence(self, parent_id: str) -> int:
        return await self.cache.aincrement(
            parent_id,
            namespace=REALTIME_INBOX_SEQUENCE_NAMESPACE,
            ttl_seconds=INBOX_RETENTION_SECONDS,
        )

    # Clear

    def _clear(self, parent_id: str) -> None:
        self.cache.clear_scored(parent_id, namespace=REALTIME_INBOX_NAMESPACE)
        self.cache.remove(parent_id, namespace=REALTIME_INBOX_SEQUENCE_NAMESPACE)

    async def _aclear(self, parent_id: str) -> None:
        await self.cache.aclear_scored(parent_id, namespace=REALTIME_INBOX_NAMESPACE)
        await self.cache.aremove(parent_id, namespace=REALTIME_INBOX_SEQUENCE_NAMESPACE)

    # Replace range

    def _replace_range(
        self,
        parent_id: str,
        *,
        up_to_sequence: int,
        replacements: list[InboxEntry],
    ) -> list[InboxEntry]:
        raise NotImplementedError("A durable inbox is append-only")

    async def _areplace_range(
        self,
        parent_id: str,
        *,
        up_to_sequence: int,
        replacements: list[InboxEntry],
    ) -> list[InboxEntry]:
        raise NotImplementedError("A durable inbox is append-only")


__all__ = ["CacheRealtimeInboxStore"]
