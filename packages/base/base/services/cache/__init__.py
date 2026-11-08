from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Never
from uuid import UUID, uuid5

from base.models.definitions.cache import CacheChannels, CacheKeys
from base.models.view_models.exceptions.cache import (
    CacheBadRequestError,
    CacheUnexpectedError,
)
from base.tools.serializers import serialize


class AbstractCacheService(ABC):
    """Abstract class for cache service."""

    # Basic Get/Set

    @abstractmethod
    def get(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value_type: Any = str,
    ) -> Any:
        """Gets data from the cache using key and namespace (optional)."""

    @abstractmethod
    async def aget(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value_type: Any = str,
    ) -> Any:
        """Asynchronously gets data from the cache using key and namespace (optional)."""

    @abstractmethod
    def set(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Stores data in the cache using key and namespace (optional)."""

    @abstractmethod
    async def aset(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Asynchronously stores data in the cache using key and namespace (optional)."""

    @abstractmethod
    def remove(self, key: CacheKeys | str, namespace: UUID | str | None = None) -> None:
        """Removes data from the cache using key and namespace (optional)."""

    @abstractmethod
    async def aremove(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        """Asynchronously removes data from the cache using key and namespace (optional)."""

    # Hashset Operations

    @abstractmethod
    def get_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> dict | None:
        """Gets hash data from the cache using key and namespace (optional)."""

    @abstractmethod
    async def aget_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> dict | None:
        """Asynchronously gets hash data from the cache using key and namespace (optional)."""

    @abstractmethod
    def delete_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> None:
        """Deletes hash data from the cache using key and namespace (optional)."""

    @abstractmethod
    async def adelete_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> None:
        """Asynchronously deletes hash data from the cache using key and namespace (optional)."""

    @abstractmethod
    def get_hashset_all(
        self, key: CacheKeys | str, namespace: UUID | None = None
    ) -> dict | None:
        """Gets all hash data from the cache using key and namespace (optional)."""

    @abstractmethod
    async def aget_hashset_all(
        self, key: CacheKeys | str, namespace: UUID | None = None
    ) -> dict | None:
        """Asynchronously gets all hash data from the cache using key and namespace (optional)."""

    @abstractmethod
    def set_hashset(
        self,
        key: CacheKeys | str,
        namespace: UUID | None = None,
        hashset: dict | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Stores hash data in the cache using key and namespace (optional)."""

    @abstractmethod
    async def aset_hashset(
        self,
        key: CacheKeys | str,
        namespace: UUID | None = None,
        hashset: dict | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Asynchronously stores hash data in the cache using key and namespace (optional)."""

    # List Operations

    @abstractmethod
    def append_to_list(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int = 600,
    ) -> None:
        """Appends a value to a list in the cache."""

    @abstractmethod
    async def aappend_to_list(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int = 600,
    ) -> None:
        """Asynchronously appends a value to a list in the cache."""

    @abstractmethod
    def get_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> list[Any]:
        """Gets a list from the cache."""

    @abstractmethod
    async def aget_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> list[Any]:
        """Asynchronously gets a list from the cache."""

    @abstractmethod
    async def aget_list_raw(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> list[str]:
        """Asynchronously gets a list from the cache without any deserialization."""

    @abstractmethod
    def clear_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        """Clears a list from the cache."""

    @abstractmethod
    async def aclear_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        """Asynchronously clears a list from the cache."""

    # Distributed Locks

    @abstractmethod
    def acquire_lock(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        ttl_seconds: int = 30,
    ) -> bool:
        """Acquires a distributed lock. Returns True if the lock was acquired, False otherwise."""

    @abstractmethod
    async def aacquire_lock(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        ttl_seconds: int = 30,
    ) -> bool:
        """Asynchronously acquires a distributed lock."""

    @abstractmethod
    def release_lock(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        """Releases a distributed lock."""

    @abstractmethod
    async def arelease_lock(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        """Asynchronously releases a distributed lock."""

    # Pub/Sub

    @abstractmethod
    def publish(
        self, channel: CacheChannels | str, message: str, ttl_seconds: int = 86400
    ) -> int:
        """Publish a message to a channel. Returns number of receivers."""

    @abstractmethod
    async def apublish(
        self, channel: CacheChannels | str, message: str, ttl_seconds: int = 86400
    ) -> int:
        """Asynchronously publish a message to a channel. Returns number of receivers."""

    @abstractmethod
    def subscribe(self, channel: CacheChannels | str) -> AsyncGenerator[str, None]:
        """Subscribe to a channel. Returns an async generator that yields messages."""

    @abstractmethod
    def publish_batch(
        self,
        channel: CacheChannels | str,
        messages: list[str],
        ttl_seconds: int = 86400,
    ) -> None:
        """Publish a batch of messages to a channel."""

    @abstractmethod
    async def apublish_batch(
        self,
        channel: CacheChannels | str,
        messages: list[str],
        ttl_seconds: int = 86400,
    ) -> None:
        """Asynchronously publish a batch of messages to a channel."""

    # Misc

    @abstractmethod
    def get_service_name(self) -> str:
        """Returns the name of the cache service."""

    # Internals

    @staticmethod
    def _check_ttl(ttl_seconds: int | None) -> None:
        """Refuse an expiry that is not in the future, as Redis itself does."""
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise CacheBadRequestError("invalid expire time")

    @staticmethod
    def _encode(value: Any) -> str:
        """Encode a value for storage, reporting a refusal as a cache error."""
        try:
            return serialize(value)
        except TypeError as exc:
            raise CacheUnexpectedError(str(exc), provider_error=exc) from exc

    @staticmethod
    def _refuse_type_clash() -> Never:
        """One key holds one kind of value, whichever backend is behind this."""
        raise CacheBadRequestError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )

    @staticmethod
    def _resolve_channel(channel: CacheChannels | str) -> str:
        return channel.value if isinstance(channel, CacheChannels) else channel

    @staticmethod
    def _get_key(key: CacheKeys | str, namespace: UUID | str | None = None) -> str:
        """One entry per key, narrowed by the namespace it was stored under.

        A namespace that does not parse as a UUID is folded into one rather
        than replacing the key: a name used as a namespace used to return
        itself, which gave everything stored under it a single shared entry.
        """
        name = key.value if isinstance(key, CacheKeys) else str(key)
        if isinstance(namespace, str):
            try:
                namespace = UUID(namespace)
            except ValueError:
                namespace = uuid5(UUID(int=0), namespace)
        return str(uuid5(namespace or UUID(int=0), name))
