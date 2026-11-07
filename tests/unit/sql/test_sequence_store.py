"""`AbstractSequenceStore`'s contract, and the in-memory store that honours it.

The abstract half is small but load-bearing: `replace_range` validates before
it delegates, so an implementation never sees an empty replacement list or a
replacement that would collide with a retained item. The in-memory store is
the reference implementation the rest of the workspace is checked against, so
it is pinned on both paths, including under concurrent appends.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import ConversationId, StoredMessage
from base.services.store import AbstractSequenceStore
from base.services.store.chat_history.memory import InMemoryChatHistoryStore


@pytest.fixture
def conversation() -> ConversationId:
    return ConversationId(uuid.uuid4())


@pytest.fixture
def store() -> InMemoryChatHistoryStore:
    return InMemoryChatHistoryStore()


def message(
    conversation: ConversationId, sequence: int, content: str | None = None
) -> StoredMessage:
    return StoredMessage(
        conversation_id=conversation,
        sequence=sequence,
        role=MessageRole.USER,
        content=content or f"m{sequence}",
    )


def seed(
    store: InMemoryChatHistoryStore, conversation: ConversationId, count: int
) -> list[StoredMessage]:
    return store.append(
        conversation, [message(conversation, i) for i in range(1, count + 1)]
    )


# Append


def test_append_returns_the_items_it_stored(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    stored = seed(store, conversation, 3)
    assert [item.sequence for item in stored] == [1, 2, 3]


async def test_aappend_returns_the_items_it_stored(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    stored = await store.aappend(
        conversation, [message(conversation, i) for i in (1, 2)]
    )
    assert [item.sequence for item in stored] == [1, 2]


def test_append_preserves_the_order_it_was_given(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    store.append(conversation, [message(conversation, 2), message(conversation, 1)])
    assert [item.sequence for item in store.get(conversation)] == [2, 1]


def test_append_adds_to_what_is_already_there(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 2)
    store.append(conversation, [message(conversation, 3)])
    assert store.count(conversation) == 3


def test_append_of_nothing_leaves_the_parent_alone(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 2)
    assert store.append(conversation, []) == []
    assert store.count(conversation) == 2


def test_two_parents_do_not_share_a_sequence(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    other = ConversationId(uuid.uuid4())
    seed(store, conversation, 3)
    seed(store, other, 1)
    assert store.count(conversation) == 3
    assert store.count(other) == 1


# Get


def test_get_returns_everything_when_no_limit_is_asked_for(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 4)
    assert [item.sequence for item in store.get(conversation)] == [1, 2, 3, 4]


async def test_aget_returns_everything_when_no_limit_is_asked_for(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    await store.aappend(conversation, [message(conversation, i) for i in (1, 2, 3)])
    assert [item.sequence for item in await store.aget(conversation)] == [1, 2, 3]


def test_a_limit_takes_the_most_recent_items_in_ascending_order(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 5)
    assert [item.sequence for item in store.get(conversation, limit=2)] == [4, 5]


async def test_an_async_limit_takes_the_most_recent_items_in_ascending_order(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    await store.aappend(conversation, [message(conversation, i) for i in range(1, 6)])
    assert [item.sequence for item in await store.aget(conversation, limit=2)] == [4, 5]


def test_a_limit_larger_than_the_history_returns_all_of_it(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    assert len(store.get(conversation, limit=99)) == 3


def test_a_limit_of_zero_returns_everything(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    """A defect, pinned as it stands: `items[-0:]` is the whole list."""
    seed(store, conversation, 3)
    assert len(store.get(conversation, limit=0)) == 3


def test_a_parent_with_no_items_reads_as_empty(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    assert store.get(conversation) == []
    assert store.get(conversation, limit=5) == []


def test_get_hands_back_a_copy_rather_than_the_stored_list(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 2)
    store.get(conversation).clear()
    assert store.count(conversation) == 2


# Count


def test_count_is_zero_for_a_parent_that_was_never_written(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    assert store.count(conversation) == 0


async def test_acount_is_zero_for_a_parent_that_was_never_written(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    assert await store.acount(conversation) == 0


def test_count_follows_what_was_appended(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 4)
    assert store.count(conversation) == 4


async def test_acount_follows_what_was_appended(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    await store.aappend(conversation, [message(conversation, 1)])
    assert await store.acount(conversation) == 1


# Next sequence


def test_the_first_sequence_is_one(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    assert store.next_sequence(conversation) == 1


async def test_the_first_async_sequence_is_one(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    assert await store.anext_sequence(conversation) == 1


def test_the_next_sequence_follows_the_highest_stored_one(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    assert store.next_sequence(conversation) == 4


def test_the_next_sequence_ignores_gaps_below_the_highest(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    store.append(conversation, [message(conversation, 1), message(conversation, 9)])
    assert store.next_sequence(conversation) == 10


async def test_the_next_async_sequence_follows_the_highest_stored_one(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    await store.aappend(conversation, [message(conversation, i) for i in (1, 2)])
    assert await store.anext_sequence(conversation) == 3


def test_next_sequence_is_per_parent(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    other = ConversationId(uuid.uuid4())
    seed(store, conversation, 5)
    assert store.next_sequence(other) == 1


# Clear


def test_clear_empties_the_parent(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    store.clear(conversation)
    assert store.count(conversation) == 0
    assert store.next_sequence(conversation) == 1


async def test_aclear_empties_the_parent(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    await store.aappend(conversation, [message(conversation, 1)])
    await store.aclear(conversation)
    assert await store.acount(conversation) == 0


def test_clearing_twice_is_not_an_error(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 1)
    store.clear(conversation)
    store.clear(conversation)
    assert store.count(conversation) == 0


def test_clearing_one_parent_leaves_the_others(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    other = ConversationId(uuid.uuid4())
    seed(store, conversation, 2)
    seed(store, other, 2)
    store.clear(conversation)
    assert store.count(other) == 2


# Replace range


def test_replace_range_swaps_the_items_up_to_the_boundary(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 4)
    returned = store.replace_range(
        conversation,
        up_to_sequence=2,
        replacements=[message(conversation, 1, "summary")],
    )
    assert [item.content for item in returned] == ["summary"]
    assert [item.content for item in store.get(conversation)] == [
        "summary",
        "m3",
        "m4",
    ]


async def test_areplace_range_swaps_the_items_up_to_the_boundary(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    await store.aappend(conversation, [message(conversation, i) for i in range(1, 5)])
    await store.areplace_range(
        conversation,
        up_to_sequence=2,
        replacements=[message(conversation, 1, "summary")],
    )
    assert [item.content for item in await store.aget(conversation)] == [
        "summary",
        "m3",
        "m4",
    ]


def test_replace_range_keeps_everything_above_the_boundary(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 5)
    store.replace_range(
        conversation, up_to_sequence=3, replacements=[message(conversation, 3, "s")]
    )
    assert [item.sequence for item in store.get(conversation)] == [3, 4, 5]


def test_replace_range_on_a_parent_with_no_items_just_inserts(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    store.replace_range(
        conversation, up_to_sequence=1, replacements=[message(conversation, 1)]
    )
    assert store.count(conversation) == 1


def test_replace_range_refuses_an_empty_replacement_list(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 2)
    with pytest.raises(ValueError, match="use clear\\(\\)"):
        store.replace_range(conversation, up_to_sequence=1, replacements=[])


async def test_areplace_range_refuses_an_empty_replacement_list(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    with pytest.raises(ValueError, match="use clear\\(\\)"):
        await store.areplace_range(conversation, up_to_sequence=1, replacements=[])


def test_replace_range_refuses_a_replacement_above_the_boundary(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    with pytest.raises(ValueError, match=r"replacements\[1\].sequence \(5\)"):
        store.replace_range(
            conversation,
            up_to_sequence=2,
            replacements=[message(conversation, 1), message(conversation, 5)],
        )


async def test_areplace_range_refuses_a_replacement_above_the_boundary(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    with pytest.raises(ValueError, match=r"replacements\[0\].sequence \(5\)"):
        await store.areplace_range(
            conversation, up_to_sequence=2, replacements=[message(conversation, 5)]
        )


def test_a_refused_replace_range_writes_nothing(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    """Validation runs in the wrapper, before the implementation is reached."""
    seed(store, conversation, 2)
    with pytest.raises(ValueError):
        store.replace_range(
            conversation, up_to_sequence=1, replacements=[message(conversation, 9)]
        )
    assert [item.sequence for item in store.get(conversation)] == [1, 2]


def test_a_replacement_exactly_at_the_boundary_is_allowed(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    store.replace_range(
        conversation, up_to_sequence=2, replacements=[message(conversation, 2, "s")]
    )
    assert [item.content for item in store.get(conversation)] == ["s", "m3"]


def test_the_validator_is_reachable_without_a_store() -> None:
    conversation = ConversationId(uuid.uuid4())
    AbstractSequenceStore._validate_replacements(3, [message(conversation, 3)])


# Concurrency


def test_two_threads_appending_lose_nothing(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    """The per-parent lock is what stops one thread's extend from losing another's."""
    barrier = threading.Barrier(2)

    def append(offset: int) -> None:
        barrier.wait()
        for i in range(50):
            store.append(conversation, [message(conversation, offset + i)])

    threads = [threading.Thread(target=append, args=(offset,)) for offset in (1, 1000)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.count(conversation) == 100
    assert len({item.sequence for item in store.get(conversation)}) == 100


async def test_two_coroutines_appending_lose_nothing(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    async def append(offset: int) -> None:
        for i in range(50):
            await store.aappend(conversation, [message(conversation, offset + i)])
            await asyncio.sleep(0)

    await asyncio.gather(append(1), append(1000))

    assert await store.acount(conversation) == 100
    assert len({item.sequence for item in await store.aget(conversation)}) == 100


async def test_sequences_handed_out_under_concurrency_stay_usable(
    store: InMemoryChatHistoryStore, conversation: ConversationId
) -> None:
    """Appending what `next_sequence` just returned never reuses a number."""
    lock = asyncio.Lock()

    async def append_one() -> None:
        async with lock:
            sequence = await store.anext_sequence(conversation)
            await store.aappend(conversation, [message(conversation, sequence)])

    await asyncio.gather(*(append_one() for _ in range(20)))

    sequences = [item.sequence for item in await store.aget(conversation)]
    assert sorted(sequences) == list(range(1, 21))


def test_the_store_is_one_instance_per_process(store: Any) -> None:
    assert InMemoryChatHistoryStore() is store
