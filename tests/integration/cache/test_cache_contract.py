"""The contract both cache backends answer to.

`get_cache_service` hands a service back behind `AbstractCacheService`, so a
caller cannot know which one it has. Everything it is entitled to assume is
pinned here and run against both, because a difference between them is a bug
that only shows up in whichever environment runs the other backend.
"""

import time
from typing import Any
from uuid import UUID

import pytest

from base.models.definitions.cache import CacheKeys
from base.services.cache import AbstractCacheService

ROUND_TRIP_VALUES: list[Any] = [
    pytest.param({"a": 1, "b": [1, 2]}, id="dict"),
    pytest.param([1, "two", 3.0, None], id="list"),
    pytest.param(7, id="int"),
    pytest.param(-7, id="negative-int"),
    pytest.param(0, id="zero"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param({"deep": {"deeper": [{"deepest": "é🎉日本語"}]}}, id="nested-unicode"),
    pytest.param({"blob": "x" * 200_000}, id="large"),
    pytest.param({}, id="empty-dict"),
    pytest.param([], id="empty-list"),
]


@pytest.mark.parametrize("value", ROUND_TRIP_VALUES)
async def test_a_value_returns_as_it_was_stored(
    cache: AbstractCacheService, namespace: UUID, value: Any
) -> None:
    await cache.aset("round-trip", namespace, value=value)
    assert await cache.aget("round-trip", namespace) == value


@pytest.mark.parametrize("value", ROUND_TRIP_VALUES)
def test_a_value_returns_as_it_was_stored_synchronously(
    cache: AbstractCacheService, namespace: UUID, value: Any
) -> None:
    cache.set("round-trip-sync", namespace, value=value)
    assert cache.get("round-trip-sync", namespace) == value


async def test_a_value_written_synchronously_is_read_asynchronously(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    cache.set("crossover", namespace, value={"written": "sync"})
    assert await cache.aget("crossover", namespace) == {"written": "sync"}


async def test_a_missing_key_reads_as_none(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    assert await cache.aget("never-written", namespace) is None


async def test_a_stored_none_is_indistinguishable_from_a_miss(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    """Pins a limit of the contract: `None` cannot be cached as a known value.

    `serialize(None)` is `"null"` and both backends decode that back to `None`,
    which is also what a missing key returns. A caller caching "we looked and
    there was nothing" cannot tell it apart from "we never looked".
    """
    await cache.aset("explicit-none", namespace, value=None)
    assert await cache.aget("explicit-none", namespace) is None
    assert await cache.aget("never-written", namespace) is None


async def test_writing_again_replaces_the_value(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset("mutable", namespace, value={"v": 1})
    await cache.aset("mutable", namespace, value={"v": 2})
    assert await cache.aget("mutable", namespace) == {"v": 2}


async def test_the_same_key_in_two_namespaces_holds_two_values(
    cache: AbstractCacheService, namespace: UUID, other_namespace: UUID
) -> None:
    await cache.aset("shared", namespace, value={"owner": "first"})
    await cache.aset("shared", other_namespace, value={"owner": "second"})
    assert await cache.aget("shared", namespace) == {"owner": "first"}
    assert await cache.aget("shared", other_namespace) == {"owner": "second"}


async def test_two_keys_in_one_namespace_hold_two_values(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset("first", namespace, value={"n": 1})
    await cache.aset("second", namespace, value={"n": 2})
    assert await cache.aget("first", namespace) == {"n": 1}
    assert await cache.aget("second", namespace) == {"n": 2}


async def test_a_namespace_given_as_text_addresses_the_same_entry(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset("addressed", namespace, value={"n": 1})
    assert await cache.aget("addressed", str(namespace)) == {"n": 1}


async def test_an_entry_without_a_namespace_is_not_the_namespaced_one(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset("global", None, value={"scope": "none"})
    await cache.aset("global", namespace, value={"scope": "namespaced"})
    assert await cache.aget("global", None) == {"scope": "none"}
    assert await cache.aget("global", namespace) == {"scope": "namespaced"}


async def test_an_enum_key_and_its_text_address_the_same_entry(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset(CacheKeys.TEST, namespace, value={"via": "enum"})
    assert await cache.aget(CacheKeys.TEST.value, namespace) == {"via": "enum"}


async def test_a_ttl_expires_the_entry(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset("brief", namespace, value={"a": 1}, ttl_seconds=1)
    assert await cache.aget("brief", namespace) == {"a": 1}
    time.sleep(1.3)
    assert await cache.aget("brief", namespace) is None


async def test_no_ttl_leaves_the_entry_in_place(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset("persistent", namespace, value={"a": 1}, ttl_seconds=None)
    time.sleep(1.3)
    assert await cache.aget("persistent", namespace) == {"a": 1}


async def test_removing_an_entry_makes_it_a_miss(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset("doomed", namespace, value={"a": 1})
    await cache.aremove("doomed", namespace)
    assert await cache.aget("doomed", namespace) is None


async def test_removing_an_entry_that_was_never_written_is_accepted(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aremove("never-written", namespace)
    assert await cache.aget("never-written", namespace) is None


async def test_removing_twice_is_accepted(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset("doomed", namespace, value={"a": 1})
    await cache.aremove("doomed", namespace)
    await cache.aremove("doomed", namespace)
    assert await cache.aget("doomed", namespace) is None


def test_removing_synchronously_makes_it_a_miss(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    cache.set("doomed-sync", namespace, value={"a": 1})
    cache.remove("doomed-sync", namespace)
    assert cache.get("doomed-sync", namespace) is None


async def test_removing_one_namespace_leaves_the_other_alone(
    cache: AbstractCacheService, namespace: UUID, other_namespace: UUID
) -> None:
    await cache.aset("shared", namespace, value={"owner": "first"})
    await cache.aset("shared", other_namespace, value={"owner": "second"})
    await cache.aremove("shared", namespace)
    assert await cache.aget("shared", other_namespace) == {"owner": "second"}


async def test_hashset_fields_return_decoded(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset_hashset(
        "profile", namespace, hashset={"name": '"ada"', "age": "36"}
    )
    assert await cache.aget_hashset_all("profile", namespace) == {
        "name": "ada",
        "age": 36,
    }


def test_hashset_fields_return_decoded_synchronously(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    cache.set_hashset("profile-sync", namespace, hashset={"name": '"ada"'})
    assert cache.get_hashset_all("profile-sync", namespace) == {"name": "ada"}


async def test_named_hashset_fields_return_in_the_order_asked_for(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset_hashset(
        "profile", namespace, hashset={"name": '"ada"', "age": "36"}
    )
    assert await cache.aget_hashset_pairs("profile", namespace, "age", "name") == {
        "age": 36,
        "name": "ada",
    }


def test_named_hashset_fields_return_synchronously(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    cache.set_hashset("profile-sync", namespace, hashset={"age": "36"})
    assert cache.get_hashset_pairs("profile-sync", namespace, "age") == {"age": 36}


async def test_a_field_that_is_not_json_returns_as_its_text(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset_hashset("profile", namespace, hashset={"note": "plain text"})
    assert await cache.aget_hashset_pairs("profile", namespace, "note") == {
        "note": "plain text"
    }


async def test_a_missing_field_of_a_present_hashset_is_none(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset_hashset("profile", namespace, hashset={"name": '"ada"'})
    assert await cache.aget_hashset_pairs("profile", namespace, "absent") == {
        "absent": None
    }


async def test_a_hashset_that_was_never_written_reads_as_none(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    assert await cache.aget_hashset_all("never-written", namespace) is None


async def test_deleting_a_field_leaves_the_rest(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset_hashset(
        "profile", namespace, hashset={"name": '"ada"', "age": "36"}
    )
    await cache.adelete_hashset_pairs("profile", namespace, "age")
    assert await cache.aget_hashset_all("profile", namespace) == {"name": "ada"}


def test_deleting_a_field_synchronously_leaves_the_rest(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    cache.set_hashset("profile-sync", namespace, hashset={"a": "1", "b": "2"})
    cache.delete_hashset_pairs("profile-sync", namespace, "a")
    assert cache.get_hashset_all("profile-sync", namespace) == {"b": 2}


async def test_appended_items_return_in_order(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    for item in ({"n": 1}, {"n": 2}, {"n": 3}):
        await cache.aappend_to_list("events", namespace, value=item)
    assert await cache.aget_list("events", namespace) == [{"n": 1}, {"n": 2}, {"n": 3}]


def test_appended_items_return_in_order_synchronously(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    cache.append_to_list("events-sync", namespace, value={"n": 1})
    cache.append_to_list("events-sync", namespace, value={"n": 2})
    assert cache.get_list("events-sync", namespace) == [{"n": 1}, {"n": 2}]


async def test_a_list_that_was_never_written_is_empty(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    assert await cache.aget_list("never-written", namespace) == []


async def test_a_raw_list_returns_the_stored_text(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aappend_to_list("events", namespace, value={"n": 1})
    assert await cache.aget_list_raw("events", namespace) == ['{"n": 1}']


async def test_clearing_a_list_empties_it(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aappend_to_list("events", namespace, value={"n": 1})
    await cache.aclear_list("events", namespace)
    assert await cache.aget_list("events", namespace) == []


async def test_a_lock_is_held_by_whoever_took_it_first(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    assert await cache.aacquire_lock("job", namespace, ttl_seconds=30) is True
    assert await cache.aacquire_lock("job", namespace, ttl_seconds=30) is False


async def test_releasing_a_lock_lets_the_next_caller_take_it(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aacquire_lock("job", namespace, ttl_seconds=30)
    await cache.arelease_lock("job", namespace)
    assert await cache.aacquire_lock("job", namespace, ttl_seconds=30) is True


def test_a_lock_is_held_synchronously_too(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    assert cache.acquire_lock("job-sync", namespace, ttl_seconds=30) is True
    assert cache.acquire_lock("job-sync", namespace, ttl_seconds=30) is False
    cache.release_lock("job-sync", namespace)
    assert cache.acquire_lock("job-sync", namespace, ttl_seconds=30) is True


async def test_a_lock_expires_on_its_own(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    assert await cache.aacquire_lock("job", namespace, ttl_seconds=1) is True
    time.sleep(1.3)
    assert await cache.aacquire_lock("job", namespace, ttl_seconds=1) is True


async def test_locks_in_two_namespaces_are_independent(
    cache: AbstractCacheService, namespace: UUID, other_namespace: UUID
) -> None:
    assert await cache.aacquire_lock("job", namespace, ttl_seconds=30) is True
    assert await cache.aacquire_lock("job", other_namespace, ttl_seconds=30) is True


def test_the_service_names_itself(cache: AbstractCacheService) -> None:
    assert cache.get_service_name()


async def test_take_returns_the_value_and_removes_it(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aset("taken", namespace, value="ticket")
    assert await cache.atake("taken", namespace) == "ticket"
    assert await cache.aget("taken", namespace) is None


def test_take_returns_the_value_and_removes_it_synchronously(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    cache.set("taken", namespace, value="ticket")
    assert cache.take("taken", namespace) == "ticket"
    assert cache.get("taken", namespace) is None


async def test_take_of_a_missing_key_is_none(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    assert await cache.atake("absent", namespace) is None


async def test_concurrent_takes_hand_the_value_to_exactly_one_caller(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    import asyncio

    for _ in range(20):
        await cache.aset("contended", namespace, value="ticket")
        results = await asyncio.gather(
            *(cache.atake("contended", namespace) for _ in range(8))
        )
        assert sum(r is not None for r in results) == 1


async def test_increment_counts_from_one(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    assert await cache.aincrement("counter", namespace) == 1
    assert await cache.aincrement("counter", namespace) == 2


def test_increment_counts_from_one_synchronously(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    assert cache.increment("counter", namespace) == 1
    assert cache.increment("counter", namespace) == 2


async def test_an_incremented_counter_reads_back_as_a_number(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aincrement("counter", namespace)
    await cache.aincrement("counter", namespace)
    assert await cache.aget("counter", namespace, value_type=int) == 2


async def test_concurrent_increments_never_return_the_same_value(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    import asyncio

    values = await asyncio.gather(
        *(cache.aincrement("contended-counter", namespace) for _ in range(50))
    )
    assert sorted(values) == list(range(1, 51))


async def test_scored_values_come_back_in_score_order(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    for score in (3, 1, 2):
        await cache.aadd_scored("scored", namespace, value=f"v{score}", score=score)
    assert await cache.aget_scored("scored", namespace) == ["v1", "v2", "v3"]


async def test_only_values_scored_above_the_cursor_are_returned(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    for score in (1, 2, 3):
        await cache.aadd_scored("scored", namespace, value=f"v{score}", score=score)
    assert await cache.aget_scored("scored", namespace, above=1) == ["v2", "v3"]


async def test_a_cap_keeps_the_highest_scores_whatever_order_they_arrived(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    """The case a capped list got wrong: a late low score must not displace a high one."""
    for score in (1, 2, 4, 5, 3):
        await cache.aadd_scored(
            "capped", namespace, value=f"v{score}", score=score, max_length=3
        )
    assert await cache.aget_scored("capped", namespace) == ["v3", "v4", "v5"]


def test_scored_values_round_trip_synchronously(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    cache.add_scored("scored", namespace, value={"a": 1}, score=1)
    assert cache.get_scored("scored", namespace) == [{"a": 1}]
    cache.clear_scored("scored", namespace)
    assert cache.get_scored("scored", namespace) == []


async def test_a_cleared_scored_set_is_empty(
    cache: AbstractCacheService, namespace: UUID
) -> None:
    await cache.aadd_scored("scored", namespace, value="v", score=1)
    await cache.aclear_scored("scored", namespace)
    assert await cache.aget_scored("scored", namespace) == []
