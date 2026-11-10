from abc import ABC, abstractmethod
from typing import final

from base.models.view_models.realtime import InboxEntry
from base.services.store import AbstractSequenceStore


class RealtimeInboxStore(AbstractSequenceStore[str, InboxEntry], ABC):
    """Ordered durable messages per user, keyed by tenant-scoped user key."""

    @abstractmethod
    def _after(self, parent_id: str, sequence: int) -> list[InboxEntry]: ...

    @final
    def after(self, parent_id: str, sequence: int) -> list[InboxEntry]:
        """Entries strictly after `sequence`, in ascending order."""
        return self._after(parent_id, sequence)

    @abstractmethod
    async def _aafter(self, parent_id: str, sequence: int) -> list[InboxEntry]: ...

    @final
    async def aafter(self, parent_id: str, sequence: int) -> list[InboxEntry]:
        """Entries strictly after `sequence`, in ascending order, asynchronously."""
        return await self._aafter(parent_id, sequence)


__all__ = ["RealtimeInboxStore"]
