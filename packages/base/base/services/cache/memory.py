import asyncio
import inspect
import json
import threading
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any, ClassVar
from uuid import UUID

from base.models.definitions.cache import CacheChannels, CacheKeys
from base.models.view_models.exceptions.cache import (
    CacheBadRequestError,
)
from base.services.cache import AbstractCacheService
from base.services.common.meta import SingletonABC
from base.tools.deserializers import ExpansiveDecoder
from base.tools.misc import generic_deep_deserialize, parse_json_pairs


class _TTLEntry:
    """A cache entry with optional TTL tracking."""

    __slots__ = ("expires_at", "value")

    def __init__(self, value: Any, ttl_seconds: int | None = None) -> None:
        self.value: Any = value
        self.expires_at: float | None = (
            time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        )

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.monotonic() >= self.expires_at


class InMemoryCacheService(AbstractCacheService, metaclass=SingletonABC):
    SERVICE_NAME: ClassVar[str] = "In-Memory Cache"

    def __init__(self) -> None:
        super().__init__()
        self._store: dict[str, _TTLEntry] = {}
        self._hashsets: dict[str, _TTLEntry] = {}
        self._lists: dict[str, _TTLEntry] = {}
        self._scored: dict[str, _TTLEntry] = {}
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[asyncio.Queue[str]]] = defaultdict(list)

    # TTL helpers

    def _get_entry(self, store: dict[str, _TTLEntry], key: str) -> _TTLEntry | None:
        entry = store.get(key)
        if entry is None:
            return None
        if entry.expired:
            store.pop(key, None)
            return None
        return entry

    # Redis parity guards

    def _check_no_type_clash(self, store: dict[str, _TTLEntry], key: str) -> None:
        """A key already held by the other store is the clash Redis reports."""
        if self._get_entry(store, key) is not None:
            self._refuse_type_clash()

    # Basic Get/Set

    def get(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value_type: Any = str,
    ) -> Any:
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._store, composite)
        if entry is None:
            return None
        raw: str = entry.value
        value = json.loads(raw, cls=ExpansiveDecoder)
        if inspect.isclass(value_type) or value_type is dict:
            return value
        return value_type(value)

    async def aget(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value_type: Any = str,
    ) -> Any:
        return self.get(key, namespace, value_type)

    def set(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._check_ttl(ttl_seconds)
        encoded = self._encode(value)
        composite = self._get_key(key, namespace)
        with self._lock:
            self._check_no_type_clash(self._hashsets, composite)
            self._store[composite] = _TTLEntry(encoded, ttl_seconds)

    async def aset(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self.set(key, namespace, value, ttl_seconds)

    def take(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> str | None:
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._store, composite)
            self._store.pop(composite, None)
        return None if entry is None else json.loads(entry.value, cls=ExpansiveDecoder)

    async def atake(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> str | None:
        return self.take(key, namespace)

    def remove(self, key: CacheKeys | str, namespace: UUID | str | None = None) -> None:
        composite = self._get_key(key, namespace)
        with self._lock:
            self._store.pop(composite, None)

    async def aremove(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        self.remove(key, namespace)

    # Counters

    def increment(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        ttl_seconds: int | None = None,
    ) -> int:
        self._check_ttl(ttl_seconds)
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._store, composite)
            value = (json.loads(entry.value) if entry else 0) + 1
            self._store[composite] = _TTLEntry(self._encode(value), ttl_seconds)
        return value

    async def aincrement(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        ttl_seconds: int | None = None,
    ) -> int:
        return self.increment(key, namespace, ttl_seconds)

    # Sorted Sets

    def add_scored(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        score: int = 0,
        ttl_seconds: int = 600,
        max_length: int | None = None,
    ) -> None:
        composite = self._get_key(key, namespace)
        encoded = self._encode(value)
        with self._lock:
            entry = self._get_entry(self._scored, composite)
            members: dict[str, int] = dict(entry.value) if entry else {}
            members[encoded] = score
            if max_length is not None and len(members) > max_length:
                kept = sorted(members.items(), key=lambda pair: pair[1])[-max_length:]
                members = dict(kept)
            self._scored[composite] = _TTLEntry(members, ttl_seconds)

    async def aadd_scored(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        score: int = 0,
        ttl_seconds: int = 600,
        max_length: int | None = None,
    ) -> None:
        self.add_scored(key, namespace, value, score, ttl_seconds, max_length)

    def get_scored(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        above: int | None = None,
    ) -> list[Any]:
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._scored, composite)
            members: dict[str, int] = dict(entry.value) if entry else {}
        ordered = sorted(members.items(), key=lambda pair: pair[1])
        return [
            json.loads(encoded, cls=ExpansiveDecoder)
            for encoded, score in ordered
            if above is None or score > above
        ]

    async def aget_scored(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        above: int | None = None,
    ) -> list[Any]:
        return self.get_scored(key, namespace, above)

    def clear_scored(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        composite = self._get_key(key, namespace)
        with self._lock:
            self._scored.pop(composite, None)

    async def aclear_scored(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        self.clear_scored(key, namespace)

    # Hashset Operations

    def get_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> dict | None:
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._hashsets, composite)
        mapping: dict[str, str] = entry.value if entry is not None else {}
        result: list[str | None] = [mapping.get(k) for k in hashset_keys]
        if not result:
            return None
        return parse_json_pairs(hashset_keys, result)

    async def aget_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> dict | None:
        return self.get_hashset_pairs(key, namespace, *hashset_keys)

    def get_hashset_all(
        self, key: CacheKeys | str, namespace: UUID | None = None
    ) -> dict | None:
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._hashsets, composite)
        if entry is None:
            return None
        mapping: dict[str, str] = entry.value
        if not mapping:
            return None
        return {
            hash_key: generic_deep_deserialize(value) if value else None
            for hash_key, value in mapping.items()
        }

    async def aget_hashset_all(
        self, key: CacheKeys | str, namespace: UUID | None = None
    ) -> dict | None:
        return self.get_hashset_all(key, namespace)

    def set_hashset(
        self,
        key: CacheKeys | str,
        namespace: UUID | None = None,
        hashset: dict | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        if hashset is None:
            return
        if not hashset:
            raise CacheBadRequestError("wrong number of arguments: no key value pairs")
        for field, value in hashset.items():
            if not isinstance(value, (str, int, float, bytes)):
                raise CacheBadRequestError(
                    f"Invalid input of type {type(value).__name__} for field {field!r}"
                )
        composite = self._get_key(key, namespace)
        with self._lock:
            self._check_no_type_clash(self._store, composite)
            entry = self._get_entry(self._hashsets, composite)
            if entry is not None:
                entry.value.update(hashset)
                if ttl_seconds is not None:
                    entry.expires_at = time.monotonic() + ttl_seconds
            else:
                self._hashsets[composite] = _TTLEntry(dict(hashset), ttl_seconds)

    async def aset_hashset(
        self,
        key: CacheKeys | str,
        namespace: UUID | None = None,
        hashset: dict | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self.set_hashset(key, namespace, hashset, ttl_seconds)

    def delete_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> None:
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._hashsets, composite)
            if entry is not None:
                for k in hashset_keys:
                    entry.value.pop(k, None)

    async def adelete_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> None:
        self.delete_hashset_pairs(key, namespace, *hashset_keys)

    # List Operations

    def append_to_list(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int = 600,
    ) -> None:
        composite = self._get_key(key, namespace)
        serialized = self._encode(value)
        with self._lock:
            entry = self._get_entry(self._lists, composite)
            if entry is not None:
                entry.value.append(serialized)
                entry.expires_at = time.monotonic() + ttl_seconds
            else:
                self._lists[composite] = _TTLEntry([serialized], ttl_seconds)

    async def aappend_to_list(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int = 600,
    ) -> None:
        self.append_to_list(key, namespace, value, ttl_seconds)

    def get_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> list[Any]:
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._lists, composite)
        if entry is None:
            return []
        items: list[str] = entry.value
        return [json.loads(x, cls=ExpansiveDecoder) for x in items] if items else []

    async def aget_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> list[Any]:
        return self.get_list(key, namespace)

    async def aget_list_raw(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> list[str]:
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._lists, composite)
        if entry is None:
            return []
        return list(entry.value)

    def clear_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        composite = self._get_key(key, namespace)
        with self._lock:
            self._lists.pop(composite, None)

    async def aclear_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        self.clear_list(key, namespace)

    # Locks

    def acquire_lock(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        ttl_seconds: int = 30,
    ) -> bool:
        composite = self._get_key(key, namespace)
        with self._lock:
            entry = self._get_entry(self._store, composite)
            if entry is not None:
                return False
            self._store[composite] = _TTLEntry("1", ttl_seconds)
            return True

    async def aacquire_lock(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        ttl_seconds: int = 30,
    ) -> bool:
        return self.acquire_lock(key, namespace, ttl_seconds)

    def release_lock(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        self.remove(key, namespace)

    async def arelease_lock(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        self.remove(key, namespace)

    # Pub/Sub

    def publish(
        self, channel: CacheChannels | str, message: str, ttl_seconds: int = 86400
    ) -> int:
        ch = self._resolve_channel(channel)
        # Through `_get_key` like every other key here: the reader composes it
        # the same way, and a raw key could never be found again.
        history_key = self._history_key(ch)
        with self._lock:
            entry = self._get_entry(self._lists, history_key)
            if entry is not None:
                entry.value.append(message)
                entry.value = entry.value[-10_000:]
                entry.expires_at = time.monotonic() + ttl_seconds
            else:
                self._lists[history_key] = _TTLEntry([message], ttl_seconds)
            queues = list(self._subscribers.get(ch, []))
        for q in queues:
            q.put_nowait(message)
        return len(queues)

    async def apublish(
        self, channel: CacheChannels | str, message: str, ttl_seconds: int = 86400
    ) -> int:
        return self.publish(channel, message, ttl_seconds)

    async def subscribe(
        self, channel: CacheChannels | str
    ) -> AsyncGenerator[str, None]:
        ch = self._resolve_channel(channel)
        queue: asyncio.Queue[str] = asyncio.Queue()
        with self._lock:
            self._subscribers[ch].append(queue)
        try:
            while True:
                data = await queue.get()
                yield data
        finally:
            with self._lock:
                self._subscribers[ch].remove(queue)

    def publish_batch(
        self,
        channel: CacheChannels | str,
        messages: list[str],
        ttl_seconds: int = 86400,
    ) -> None:
        for message in messages:
            self.publish(channel, message, ttl_seconds)

    async def apublish_batch(
        self,
        channel: CacheChannels | str,
        messages: list[str],
        ttl_seconds: int = 86400,
    ) -> None:
        for message in messages:
            self.publish(channel, message, ttl_seconds)

    # Misc

    def get_service_name(self) -> str:
        return self.SERVICE_NAME
