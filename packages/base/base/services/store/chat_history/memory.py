import asyncio
import threading

from base.models.view_models.chat_history import ConversationId, StoredMessage
from base.services.common.meta import SingletonABC
from base.services.store.chat_history import ChatHistoryStore


class InMemoryChatHistoryStore(ChatHistoryStore, metaclass=SingletonABC):
    """Reference in-memory implementation of ChatHistoryStore.

    Single-process, lost on restart. Suitable for tests and local development.
    Production should use the SQL or Mongo concrete stores.

    Unlike the SQL and Mongo stores, this implementation does not assign
    primary keys or track persistence timestamps. StoredMessage objects are
    stored as-is, with `id` remaining None unless the caller sets it. Tests
    that need populated IDs should use the SQL store or construct drafts
    with an explicit ID.

    Locking model (per-conversation, sync and async independent):

      - Sync methods acquire a per-conversation threading.Lock.
      - Async methods acquire a per-conversation asyncio.Lock.
      - The two lock maps are independent. Sync and async callers do not
        coordinate with each other — a process should use one path or the
        other, not both, against the same conversation. Real stores do not
        have this limitation because they serialize through the database;
        the reference store accepts it as a documented constraint.

      - Lock dicts are themselves protected by a brief "lock-of-locks"
        during lookup/creation only. The per-conversation lock is held
        for the duration of the operation; the lock-of-locks is released
        before the operation runs.
    """

    def __init__(self) -> None:
        self._data: dict[ConversationId, list[StoredMessage]] = {}

        self._sync_locks: dict[ConversationId, threading.Lock] = {}
        self._sync_locks_guard: threading.Lock = threading.Lock()

        self._async_locks: dict[ConversationId, asyncio.Lock] = {}
        self._async_locks_guard: asyncio.Lock = asyncio.Lock()

    # Lock resolution

    def _sync_lock_for(self, parent_id: ConversationId) -> threading.Lock:
        with self._sync_locks_guard:
            lock = self._sync_locks.get(parent_id)
            if lock is None:
                lock = threading.Lock()
                self._sync_locks[parent_id] = lock
            return lock

    async def _async_lock_for(self, parent_id: ConversationId) -> asyncio.Lock:
        async with self._async_locks_guard:
            lock = self._async_locks.get(parent_id)
            if lock is None:
                lock = asyncio.Lock()
                self._async_locks[parent_id] = lock
            return lock

    # Append

    def _append(
        self,
        parent_id: ConversationId,
        items: list[StoredMessage],
    ) -> list[StoredMessage]:
        with self._sync_lock_for(parent_id):
            bucket = self._data.setdefault(parent_id, [])
            bucket.extend(items)
        return items

    async def _aappend(
        self,
        parent_id: ConversationId,
        items: list[StoredMessage],
    ) -> list[StoredMessage]:
        lock = await self._async_lock_for(parent_id)
        async with lock:
            bucket = self._data.setdefault(parent_id, [])
            bucket.extend(items)
        return items

    # Get

    def _get(
        self,
        parent_id: ConversationId,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        with self._sync_lock_for(parent_id):
            items = list(self._data.get(parent_id, ()))
        if limit is not None and len(items) > limit:
            items = items[-limit:]
        return items

    async def _aget(
        self,
        parent_id: ConversationId,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        lock = await self._async_lock_for(parent_id)
        async with lock:
            items = list(self._data.get(parent_id, ()))
        if limit is not None and len(items) > limit:
            items = items[-limit:]
        return items

    # Count

    def _count(self, parent_id: ConversationId) -> int:
        with self._sync_lock_for(parent_id):
            return len(self._data.get(parent_id, ()))

    async def _acount(self, parent_id: ConversationId) -> int:
        lock = await self._async_lock_for(parent_id)
        async with lock:
            return len(self._data.get(parent_id, ()))

    # Next sequence

    def _next_sequence(self, parent_id: ConversationId) -> int:
        with self._sync_lock_for(parent_id):
            items = self._data.get(parent_id, ())
            if not items:
                return 1
            return max(m.sequence for m in items) + 1

    async def _anext_sequence(self, parent_id: ConversationId) -> int:
        lock = await self._async_lock_for(parent_id)
        async with lock:
            items = self._data.get(parent_id, ())
            if not items:
                return 1
            return max(m.sequence for m in items) + 1

    # Clear

    def _clear(self, parent_id: ConversationId) -> None:
        with self._sync_lock_for(parent_id):
            self._data.pop(parent_id, None)

    async def _aclear(self, parent_id: ConversationId) -> None:
        lock = await self._async_lock_for(parent_id)
        async with lock:
            self._data.pop(parent_id, None)

    # Replace range

    def _replace_range(
        self,
        parent_id: ConversationId,
        *,
        up_to_sequence: int,
        replacements: list[StoredMessage],
    ) -> list[StoredMessage]:
        with self._sync_lock_for(parent_id):
            items = self._data.get(parent_id, [])
            retained = [m for m in items if m.sequence > up_to_sequence]
            self._data[parent_id] = [*replacements, *retained]
        return replacements

    async def _areplace_range(
        self,
        parent_id: ConversationId,
        *,
        up_to_sequence: int,
        replacements: list[StoredMessage],
    ) -> list[StoredMessage]:
        lock = await self._async_lock_for(parent_id)
        async with lock:
            items = self._data.get(parent_id, [])
            retained = [m for m in items if m.sequence > up_to_sequence]
            self._data[parent_id] = [*replacements, *retained]
        return replacements
