import inspect
import json
from collections.abc import AsyncGenerator
from typing import Any, ClassVar
from uuid import UUID

from base.clients.redis_client import RedisClient
from base.models.definitions.cache import CacheChannels, CacheKeys
from base.services.cache import AbstractCacheService
from base.services.common.meta import SingletonABC
from base.services.error_handling import aretrying, retrying
from base.services.error_handling.cache.redis import RedisErrorHandler
from base.tools.deserializers import ExpansiveDecoder
from base.tools.misc import generic_deep_deserialize, parse_json_pairs


class RedisCacheService(AbstractCacheService, metaclass=SingletonABC):
    SERVICE_NAME: ClassVar[str] = "Redis Cache"

    def __init__(self) -> None:
        super().__init__()
        self._redis_client = RedisClient()

        # Basic Get/Set
        self._error_handler: RedisErrorHandler = RedisErrorHandler(retries=1)

    @retrying
    def get(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value_type: Any = str,
    ) -> Any:
        result: str | None = self._redis_client.sync_client.get(  # type: ignore
            self._get_key(key, namespace)
        )
        if result is None:
            return None
        value = json.loads(result, cls=ExpansiveDecoder)
        if inspect.isclass(value_type) or value_type is dict:
            return value
        return value_type(value)

    @aretrying
    async def aget(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value_type: Any = str,
    ) -> Any:
        result: str | None = await self._redis_client.async_client.get(  # type: ignore
            self._get_key(key, namespace)
        )
        if result is None:
            return None
        value = json.loads(result, cls=ExpansiveDecoder)
        if inspect.isclass(value_type) or value_type is dict:
            return value
        return value_type(value)

    @retrying
    def set(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._check_ttl(ttl_seconds)
        self._redis_client.sync_client.set(
            name=self._get_key(key, namespace),
            value=self._encode(value),
            ex=ttl_seconds,
        )

    @aretrying
    async def aset(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._check_ttl(ttl_seconds)
        await self._redis_client.async_client.set(  # type: ignore
            name=self._get_key(key, namespace),
            value=self._encode(value),
            ex=ttl_seconds,
        )

    @retrying
    def remove(self, key: CacheKeys | str, namespace: UUID | str | None = None) -> None:
        self._redis_client.sync_client.delete(self._get_key(key, namespace))

    @aretrying
    async def aremove(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        await self._redis_client.async_client.delete(  # type: ignore
            self._get_key(key, namespace)
        )

    # Hashset Operations

    @retrying
    def get_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> dict | None:
        result: list[str | None] = self._redis_client.sync_client.hmget(  # type: ignore
            self._get_key(key, namespace), list(hashset_keys)
        )
        if not result or len(result) != len(hashset_keys):
            return None
        return parse_json_pairs(hashset_keys, result)

    @aretrying
    async def aget_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> dict | None:
        result: list[str | None] = await self._redis_client.async_client.hmget(  # type: ignore
            self._get_key(key, namespace), list(hashset_keys)
        )
        if not result or len(result) != len(hashset_keys):
            return None
        return parse_json_pairs(hashset_keys, result)

    @retrying
    def get_hashset_all(
        self, key: CacheKeys | str, namespace: UUID | None = None
    ) -> dict | None:
        result: dict[str, str] = self._redis_client.sync_client.hgetall(  # type: ignore
            self._get_key(key, namespace)
        )
        if not result:
            return None
        return {
            hash_key: generic_deep_deserialize(value) if value else None
            for hash_key, value in result.items()
        }

    @aretrying
    async def aget_hashset_all(
        self, key: CacheKeys | str, namespace: UUID | None = None
    ) -> dict | None:
        result: dict[str, str] = await self._redis_client.async_client.hgetall(  # type: ignore
            self._get_key(key, namespace)
        )
        if not result:
            return None
        return {
            hash_key: generic_deep_deserialize(value) if value else None
            for hash_key, value in result.items()
        }

    @retrying
    def set_hashset(
        self,
        key: CacheKeys | str,
        namespace: UUID | None = None,
        hashset: dict | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        if hashset is None:
            return
        composite_key = self._get_key(key, namespace)
        self._redis_client.sync_client.hset(name=composite_key, mapping=hashset)
        if ttl_seconds is not None:
            self._redis_client.sync_client.expire(composite_key, ttl_seconds)

    @aretrying
    async def aset_hashset(
        self,
        key: CacheKeys | str,
        namespace: UUID | None = None,
        hashset: dict | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        if hashset is None:
            return
        composite_key = self._get_key(key, namespace)
        await self._redis_client.async_client.hset(name=composite_key, mapping=hashset)  # type: ignore
        if ttl_seconds is not None:
            await self._redis_client.async_client.expire(composite_key, ttl_seconds)  # type: ignore

    @retrying
    def delete_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> None:
        self._redis_client.sync_client.hdel(
            self._get_key(key, namespace), *hashset_keys
        )

    @aretrying
    async def adelete_hashset_pairs(
        self, key: CacheKeys | str, namespace: UUID | None = None, *hashset_keys: str
    ) -> None:
        await self._redis_client.async_client.hdel(  # type: ignore
            self._get_key(key, namespace), *hashset_keys
        )

    # List Operations

    @retrying
    def append_to_list(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int = 600,
    ) -> None:
        composite_key = self._get_key(key, namespace)
        self._redis_client.sync_client.rpush(composite_key, self._encode(value))
        self._redis_client.sync_client.expire(composite_key, ttl_seconds)

    @aretrying
    async def aappend_to_list(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        value: Any = None,
        ttl_seconds: int = 600,
    ) -> None:
        composite_key = self._get_key(key, namespace)
        await self._redis_client.async_client.rpush(composite_key, self._encode(value))  # type: ignore
        await self._redis_client.async_client.expire(composite_key, ttl_seconds)  # type: ignore

    @retrying
    def get_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> list[Any]:
        items: list[str] = self._redis_client.sync_client.lrange(  # type: ignore
            self._get_key(key, namespace), 0, -1
        )
        return [json.loads(x, cls=ExpansiveDecoder) for x in items] if items else []

    @aretrying
    async def aget_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> list[Any]:
        items: list[str] = await self._redis_client.async_client.lrange(  # type: ignore
            self._get_key(key, namespace), 0, -1
        )
        return [json.loads(x, cls=ExpansiveDecoder) for x in items] if items else []

    @aretrying
    async def aget_list_raw(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> list[str]:
        items: list[bytes | str] = await self._redis_client.async_client.lrange(  # type: ignore
            self._get_key(key, namespace), 0, -1
        )
        return (
            [x.decode() if isinstance(x, bytes) else str(x) for x in items]
            if items
            else []
        )

    @retrying
    def clear_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        self._redis_client.sync_client.delete(self._get_key(key, namespace))

    @aretrying
    async def aclear_list(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        await self._redis_client.async_client.delete(  # type: ignore
            self._get_key(key, namespace)
        )

    # Distributed Locks

    @retrying
    def acquire_lock(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        ttl_seconds: int = 30,
    ) -> bool:
        return bool(
            self._redis_client.sync_client.set(
                self._get_key(key, namespace), "1", nx=True, ex=ttl_seconds
            )
        )

    @aretrying
    async def aacquire_lock(
        self,
        key: CacheKeys | str,
        namespace: UUID | str | None = None,
        ttl_seconds: int = 30,
    ) -> bool:
        result = await self._redis_client.async_client.set(  # type: ignore
            self._get_key(key, namespace), "1", nx=True, ex=ttl_seconds
        )
        return bool(result)

    @retrying
    def release_lock(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        self._redis_client.sync_client.delete(self._get_key(key, namespace))

    @aretrying
    async def arelease_lock(
        self, key: CacheKeys | str, namespace: UUID | str | None = None
    ) -> None:
        await self._redis_client.async_client.delete(  # type: ignore
            self._get_key(key, namespace)
        )

    # Pub/Sub

    @retrying
    def publish(
        self, channel: CacheChannels | str, message: str, ttl_seconds: int = 86400
    ) -> int:
        ch = self._resolve_channel(channel)
        pipe = self._redis_client.sync_client.pipeline()
        pipe.publish(ch, message)
        pipe.rpush(f"{ch}:history", message)
        pipe.ltrim(f"{ch}:history", -10_000, -1)
        pipe.expire(f"{ch}:history", ttl_seconds)
        results: list[Any] = pipe.execute()  # type: ignore
        return results[0]

    @aretrying
    async def apublish(
        self, channel: CacheChannels | str, message: str, ttl_seconds: int = 86400
    ) -> int:
        ch = self._resolve_channel(channel)
        pipe = self._redis_client.async_client.pipeline()
        pipe.publish(ch, message)
        pipe.rpush(f"{ch}:history", message)
        pipe.ltrim(f"{ch}:history", -10_000, -1)
        pipe.expire(f"{ch}:history", ttl_seconds)
        results: list[Any] = await pipe.execute()  # type: ignore
        return results[0]

    async def subscribe(
        self, channel: CacheChannels | str
    ) -> AsyncGenerator[str, None]:
        ch = self._resolve_channel(channel)
        pubsub = self._redis_client.async_client.pubsub()
        await pubsub.subscribe(ch)
        try:
            async for msg in pubsub.listen():
                if msg["type"] != "message":
                    continue
                data = msg["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                yield data
        finally:
            await pubsub.unsubscribe(ch)
            await pubsub.close()

    @retrying
    def publish_batch(
        self,
        channel: CacheChannels | str,
        messages: list[str],
        ttl_seconds: int = 86400,
    ) -> None:
        ch = self._resolve_channel(channel)
        pipe = self._redis_client.sync_client.pipeline()
        for message in messages:
            pipe.publish(ch, message)
            pipe.rpush(f"{ch}:history", message)
        pipe.ltrim(f"{ch}:history", -10_000, -1)
        pipe.expire(f"{ch}:history", ttl_seconds)
        pipe.execute()

    @aretrying
    async def apublish_batch(
        self,
        channel: CacheChannels | str,
        messages: list[str],
        ttl_seconds: int = 86400,
    ) -> None:
        ch = self._resolve_channel(channel)
        pipe = self._redis_client.async_client.pipeline()
        for message in messages:
            pipe.publish(ch, message)
            pipe.rpush(f"{ch}:history", message)
        pipe.ltrim(f"{ch}:history", -10_000, -1)
        pipe.expire(f"{ch}:history", ttl_seconds)
        await pipe.execute()  # type: ignore

    # Misc

    def get_service_name(self) -> str:
        return self.SERVICE_NAME
