"""Behaviour the in-process cache owns on its own.

Everything the Redis service also promises is pinned by the contract suite in
`tests/integration/cache`. What is left here is what only this backend can
show cheaply: the TTL arithmetic, the pub/sub fan-out and the places where it
answers differently from Redis.
"""

import asyncio
import time
from uuid import UUID

from base.services.cache.memory import InMemoryCacheService


async def test_a_zero_ttl_entry_is_already_expired(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    """Redis refuses `ex=0`; in-process treats it as expired on arrival."""
    await memory_cache.aset("doomed", namespace, value={"a": 1}, ttl_seconds=0)
    assert await memory_cache.aget("doomed", namespace) is None


async def test_an_expired_entry_is_dropped_from_the_store(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset("doomed", namespace, value={"a": 1}, ttl_seconds=0)
    assert await memory_cache.aget("doomed", namespace) is None
    assert memory_cache._get_key("doomed", namespace) not in memory_cache._store


async def test_a_negative_ttl_entry_is_already_expired(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset("doomed", namespace, value={"a": 1}, ttl_seconds=-5)
    assert await memory_cache.aget("doomed", namespace) is None


async def test_a_short_ttl_expires_once_the_boundary_passes(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset("brief", namespace, value={"a": 1}, ttl_seconds=0.2)  # type: ignore[arg-type]
    assert await memory_cache.aget("brief", namespace) == {"a": 1}
    time.sleep(0.3)
    assert await memory_cache.aget("brief", namespace) is None


async def test_setting_again_replaces_the_expiry(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset("renewed", namespace, value={"a": 1}, ttl_seconds=0.2)  # type: ignore[arg-type]
    await memory_cache.aset("renewed", namespace, value={"a": 2})
    time.sleep(0.3)
    assert await memory_cache.aget("renewed", namespace) == {"a": 2}


async def test_an_empty_hashset_stores_nothing(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    """Redis refuses `hset` with no pairs; in-process accepts and keeps nothing."""
    await memory_cache.aset_hashset("empty", namespace, hashset={})
    assert await memory_cache.aget_hashset_all("empty", namespace) is None


async def test_a_hashset_of_none_is_a_no_op(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset_hashset("absent", namespace, hashset=None)
    assert await memory_cache.aget_hashset_all("absent", namespace) is None


async def test_pairs_from_a_missing_hashset_are_none(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    """Redis answers `{field: None}` for the same call. The two disagree."""
    assert await memory_cache.aget_hashset_pairs("absent", namespace, "field") is None


async def test_a_hashset_value_that_is_not_a_string_is_kept_as_given(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    """Redis refuses a dict field value. In-process stores the object itself."""
    await memory_cache.aset_hashset("rich", namespace, hashset={"f": {"a": 1}})
    assert await memory_cache.aget_hashset_all("rich", namespace) == {"f": {"a": 1}}


async def test_a_hashset_ttl_expires_the_whole_hash(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset_hashset(
        "brief", namespace, hashset={"f": "1"}, ttl_seconds=0
    )
    assert await memory_cache.aget_hashset_all("brief", namespace) is None


async def test_setting_a_hashset_twice_merges_the_fields(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset_hashset("merged", namespace, hashset={"a": "1"})
    await memory_cache.aset_hashset("merged", namespace, hashset={"b": "2"})
    assert await memory_cache.aget_hashset_all("merged", namespace) == {"a": 1, "b": 2}


async def test_a_hashset_and_a_value_can_share_one_key(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    """Separate stores, so neither sees the other. Redis refuses the second write."""
    await memory_cache.aset("mixed", namespace, value={"a": 1})
    await memory_cache.aset_hashset("mixed", namespace, hashset={"f": "1"})
    assert await memory_cache.aget("mixed", namespace) == {"a": 1}
    assert await memory_cache.aget_hashset_all("mixed", namespace) == {"f": 1}


async def test_an_expired_list_reads_as_empty(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aappend_to_list("gone", namespace, value={"a": 1}, ttl_seconds=0)
    assert await memory_cache.aget_list("gone", namespace) == []


async def test_a_subscriber_receives_what_is_published(
    memory_cache: InMemoryCacheService,
) -> None:
    received: list[str] = []
    stream = memory_cache.subscribe("room")

    async def drain() -> None:
        async for message in stream:
            received.append(message)
            if len(received) == 2:
                break

    task = asyncio.create_task(drain())
    await asyncio.sleep(0)
    assert await memory_cache.apublish("room", "first") == 1
    assert await memory_cache.apublish("room", "second") == 1
    await asyncio.wait_for(task, timeout=2)
    await stream.aclose()
    assert received == ["first", "second"]


async def test_publishing_with_no_subscriber_reaches_nobody(
    memory_cache: InMemoryCacheService,
) -> None:
    assert await memory_cache.apublish("quiet", "nobody home") == 0


async def test_every_subscriber_on_a_channel_receives_the_message(
    memory_cache: InMemoryCacheService,
) -> None:
    streams = [memory_cache.subscribe("room") for _ in range(3)]
    received: list[str] = []

    async def drain(stream) -> None:  # type: ignore[no-untyped-def]
        async for message in stream:
            received.append(message)
            break

    tasks = [asyncio.create_task(drain(stream)) for stream in streams]
    await asyncio.sleep(0)
    assert await memory_cache.apublish("room", "broadcast") == 3
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
    for stream in streams:
        await stream.aclose()
    assert received == ["broadcast"] * 3


async def test_a_subscriber_that_leaves_stops_counting(
    memory_cache: InMemoryCacheService,
) -> None:
    stream = memory_cache.subscribe("room")

    async def drain() -> None:
        async for _ in stream:
            break

    task = asyncio.create_task(drain())
    await asyncio.sleep(0)
    await memory_cache.apublish("room", "first")
    await asyncio.wait_for(task, timeout=2)
    await stream.aclose()
    assert await memory_cache.apublish("room", "second") == 0


async def test_published_messages_are_kept_as_channel_history(
    memory_cache: InMemoryCacheService,
) -> None:
    await memory_cache.apublish_batch("room", ["one", "two"])
    assert memory_cache._lists["room:history"].value == ["one", "two"]


def test_the_service_names_itself(memory_cache: InMemoryCacheService) -> None:
    assert memory_cache.get_service_name() == "In-Memory Cache"
