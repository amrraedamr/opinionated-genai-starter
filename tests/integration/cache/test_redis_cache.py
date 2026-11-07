"""Where the Redis cache answers differently from the in-process one.

Each test here is a divergence from the contract next door. A caller that
holds an `AbstractCacheService` cannot tell which backend it has, so every one
of these is a behaviour change waiting for whichever environment runs the
other backend.
"""

import time
from uuid import UUID

import pytest

from base.clients.redis_client import RedisClient
from base.models.view_models.exceptions.cache import (
    CacheBadRequestError,
    CacheUnexpectedError,
)
from base.services.cache.redis import RedisCacheService

pytestmark = pytest.mark.integration


async def test_a_zero_ttl_is_refused_by_the_server(
    redis_cache: RedisCacheService, namespace: UUID
) -> None:
    """In-process stores it and treats it as already expired."""
    with pytest.raises(CacheBadRequestError, match="invalid expire time"):
        await redis_cache.aset("doomed", namespace, value={"a": 1}, ttl_seconds=0)


async def test_an_empty_string_reads_back_as_a_miss(
    redis_cache: RedisCacheService, namespace: UUID
) -> None:
    """`if not result` cannot tell `""` from a missing key.

    The in-process backend raises a `JSONDecodeError` for the same value, so a
    caller caching empty strings gets a different answer per backend.
    `base/services/cache/redis.py:44`.
    """
    await redis_cache.aset("blank", namespace, value="")
    assert await redis_cache.aget("blank", namespace) is None


async def test_a_plain_string_fails_as_a_cache_error(
    redis_cache: RedisCacheService, namespace: UUID
) -> None:
    """The in-process backend raises the raw `json.JSONDecodeError` instead."""
    await redis_cache.aset("greeting", namespace, value="hello")
    with pytest.raises(CacheUnexpectedError, match="Expecting value"):
        await redis_cache.aget("greeting", namespace)


async def test_a_value_the_encoder_cannot_reach_fails_as_a_cache_error(
    redis_cache: RedisCacheService, namespace: UUID
) -> None:
    with pytest.raises(CacheUnexpectedError, match="not JSON serializable"):
        await redis_cache.aset("bad", namespace, value={1, 2})


async def test_an_empty_hashset_is_refused_by_the_server(
    redis_cache: RedisCacheService, namespace: UUID
) -> None:
    """In-process accepts the call and stores nothing."""
    with pytest.raises(CacheBadRequestError, match="no key value pairs"):
        await redis_cache.aset_hashset("empty", namespace, hashset={})


async def test_a_hashset_value_that_is_not_a_scalar_is_refused(
    redis_cache: RedisCacheService, namespace: UUID
) -> None:
    """In-process stores the object as given and hands it straight back."""
    with pytest.raises(CacheBadRequestError, match="Invalid input of type"):
        await redis_cache.aset_hashset("rich", namespace, hashset={"f": {"a": 1}})


async def test_a_hashset_cannot_share_a_key_with_a_value(
    redis_cache: RedisCacheService, namespace: UUID
) -> None:
    """One key, one Redis type. The in-process backend keeps the two apart."""
    await redis_cache.aset("mixed", namespace, value={"a": 1})
    with pytest.raises(CacheBadRequestError, match="WRONGTYPE"):
        await redis_cache.aset_hashset("mixed", namespace, hashset={"f": "1"})


async def test_pairs_from_a_missing_hashset_are_a_mapping_of_nones(
    redis_cache: RedisCacheService, namespace: UUID
) -> None:
    """In-process returns `None` for the same call."""
    assert await redis_cache.aget_hashset_pairs("absent", namespace, "field") == {
        "field": None
    }


async def test_a_hashset_ttl_expires_the_whole_hash(
    redis_cache: RedisCacheService, namespace: UUID
) -> None:
    await redis_cache.aset_hashset(
        "brief", namespace, hashset={"f": "1"}, ttl_seconds=1
    )
    assert await redis_cache.aget_hashset_all("brief", namespace) == {"f": 1}
    time.sleep(1.3)
    assert await redis_cache.aget_hashset_all("brief", namespace) is None


async def test_publishing_with_no_subscriber_reaches_nobody(
    redis_cache: RedisCacheService, unique: str
) -> None:
    channel = f"test-channel-{unique}"
    try:
        assert await redis_cache.apublish(channel, "nobody home") == 0
    finally:
        await RedisClient().async_client.delete(f"{channel}:history")


async def test_published_messages_are_kept_as_channel_history(
    redis_cache: RedisCacheService, unique: str
) -> None:
    channel = f"test-channel-{unique}"
    try:
        await redis_cache.apublish_batch(channel, ["one", "two"])
        history = await RedisClient().async_client.lrange(f"{channel}:history", 0, -1)
        assert history == ["one", "two"]
    finally:
        await RedisClient().async_client.delete(f"{channel}:history")


async def test_the_service_names_itself(redis_cache: RedisCacheService) -> None:
    assert redis_cache.get_service_name() == "Redis Cache"
