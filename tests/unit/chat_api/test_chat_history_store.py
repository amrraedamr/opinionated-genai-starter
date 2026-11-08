"""`SqlChatHistoryStore`: one row per message, ordered by a per-parent sequence.

The contract is `AbstractSequenceStore`'s, and the in-memory store in `base` is
the reference every implementation is checked against — so the expectations
here are deliberately the same ones, read against real SQL. Ordering, the
limit window and parent scoping are what a conversation's history is made of,
and each has a sync and an async implementation rather than one wrapping the
other.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import (
    ConversationId,
    StoredMessage,
    StoredToolCall,
    StoredToolCallFunction,
)
from base.models.view_models.exceptions.sql import SQLConflictError
from base.services.sql.service import SQLService
from base.tools.database import SQLResultFormats
from chat_api.models.orm.chat_history_message import ChatHistoryMessageDTO
from chat_api.stores.chat_history.sql import SqlChatHistoryStore


@pytest.fixture
def store(tables: None) -> SqlChatHistoryStore:
    return SqlChatHistoryStore()


@pytest.fixture
def conversation() -> ConversationId:
    return ConversationId(uuid.uuid4())


@pytest.fixture
def other() -> ConversationId:
    return ConversationId(uuid.uuid4())


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
    store: SqlChatHistoryStore, conversation: ConversationId, count: int
) -> list[StoredMessage]:
    return store.append(
        conversation, [message(conversation, i) for i in range(1, count + 1)]
    )


# Append


def test_append_returns_the_items_it_stored(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    stored = seed(store, conversation, 3)
    assert [item.sequence for item in stored] == [1, 2, 3]


async def test_aappend_returns_the_items_it_stored(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    stored = await store.aappend(
        conversation, [message(conversation, i) for i in (1, 2)]
    )
    assert [item.sequence for item in stored] == [1, 2]


def test_append_fills_in_the_id_the_database_assigned(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    """Drafts arrive with `id=None`; the store is what makes them real."""
    draft = message(conversation, 1)
    assert draft.id is None
    assert seed(store, conversation, 1)[0].id is not None


async def test_aappend_fills_in_the_id_the_database_assigned(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    stored = await store.aappend(conversation, [message(conversation, 1)])
    assert stored[0].id is not None


def test_append_adds_to_what_is_already_there(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 2)
    store.append(conversation, [message(conversation, 3)])
    assert store.count(conversation) == 3


def test_append_of_nothing_leaves_the_parent_alone(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 2)
    assert store.append(conversation, []) == []
    assert store.count(conversation) == 2


def test_a_sequence_already_taken_is_refused_rather_than_overwritten(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    """The unique constraint is the correctness guarantee; the store lets it through."""
    seed(store, conversation, 1)
    with pytest.raises(SQLConflictError):
        store.append(conversation, [message(conversation, 1, "collides")])


async def test_an_async_sequence_already_taken_is_refused(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 1)
    with pytest.raises(SQLConflictError):
        await store.aappend(conversation, [message(conversation, 1, "collides")])


def test_two_parents_may_hold_the_same_sequence(
    store: SqlChatHistoryStore, conversation: ConversationId, other: ConversationId
) -> None:
    seed(store, conversation, 2)
    seed(store, other, 2)
    assert (store.count(conversation), store.count(other)) == (2, 2)


# Get


def test_get_returns_everything_in_ascending_order(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    store.append(conversation, [message(conversation, s) for s in (3, 1, 2)])
    assert [item.sequence for item in store.get(conversation)] == [1, 2, 3]


async def test_aget_returns_everything_in_ascending_order(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    await store.aappend(conversation, [message(conversation, s) for s in (3, 1, 2)])
    assert [item.sequence for item in await store.aget(conversation)] == [1, 2, 3]


def test_a_limit_takes_the_most_recent_items_in_ascending_order(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 5)
    assert [item.sequence for item in store.get(conversation, limit=2)] == [4, 5]


async def test_an_async_limit_takes_the_most_recent_items_in_ascending_order(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    await store.aappend(conversation, [message(conversation, i) for i in range(1, 6)])
    assert [item.sequence for item in await store.aget(conversation, limit=2)] == [4, 5]


def test_a_limit_larger_than_the_history_returns_all_of_it(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    assert len(store.get(conversation, limit=99)) == 3


def test_a_limit_of_zero_returns_nothing(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    assert store.get(conversation, limit=0) == []


async def test_an_async_limit_of_zero_returns_nothing(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    assert await store.aget(conversation, limit=0) == []


def test_a_negative_limit_returns_nothing(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    """SQL reads `LIMIT -1` as no limit, which would hand back the whole history."""
    seed(store, conversation, 3)
    assert store.get(conversation, limit=-1) == []


async def test_an_async_negative_limit_returns_nothing(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    assert await store.aget(conversation, limit=-1) == []


def test_a_parent_with_no_items_reads_as_empty(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    assert store.get(conversation) == []
    assert store.get(conversation, limit=5) == []


def test_get_never_reaches_another_parents_messages(
    store: SqlChatHistoryStore, conversation: ConversationId, other: ConversationId
) -> None:
    seed(store, conversation, 3)
    seed(store, other, 2)
    assert [item.sequence for item in store.get(other)] == [1, 2]


def test_a_limit_is_applied_per_parent(
    store: SqlChatHistoryStore, conversation: ConversationId, other: ConversationId
) -> None:
    seed(store, conversation, 5)
    seed(store, other, 1)
    assert [item.sequence for item in store.get(other, limit=3)] == [1]


def test_a_gap_in_the_sequence_does_not_disturb_the_order(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    store.append(conversation, [message(conversation, s) for s in (1, 50, 7)])
    assert [item.sequence for item in store.get(conversation)] == [1, 7, 50]


# Count


def test_count_is_zero_for_a_parent_that_was_never_written(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    assert store.count(conversation) == 0


async def test_acount_is_zero_for_a_parent_that_was_never_written(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    assert await store.acount(conversation) == 0


def test_count_follows_what_was_appended(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 4)
    assert store.count(conversation) == 4


async def test_acount_follows_what_was_appended(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 4)
    assert await store.acount(conversation) == 4


def test_count_is_per_parent(
    store: SqlChatHistoryStore, conversation: ConversationId, other: ConversationId
) -> None:
    seed(store, conversation, 4)
    seed(store, other, 1)
    assert (store.count(conversation), store.count(other)) == (4, 1)


# Next sequence


def test_the_first_sequence_is_one(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    assert store.next_sequence(conversation) == 1


async def test_the_first_async_sequence_is_one(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    assert await store.anext_sequence(conversation) == 1


def test_the_next_sequence_follows_the_highest_stored_one(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    assert store.next_sequence(conversation) == 4


async def test_the_next_async_sequence_follows_the_highest_stored_one(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    assert await store.anext_sequence(conversation) == 4


def test_the_next_sequence_ignores_gaps_below_the_highest(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    store.append(conversation, [message(conversation, 1), message(conversation, 9)])
    assert store.next_sequence(conversation) == 10


def test_next_sequence_is_per_parent(
    store: SqlChatHistoryStore, conversation: ConversationId, other: ConversationId
) -> None:
    seed(store, conversation, 5)
    assert store.next_sequence(other) == 1


def test_a_cleared_parent_starts_counting_again(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    store.clear(conversation)
    assert store.next_sequence(conversation) == 1


# Clear


def test_clear_empties_the_parent(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    store.clear(conversation)
    assert store.count(conversation) == 0


async def test_aclear_empties_the_parent(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 3)
    await store.aclear(conversation)
    assert await store.acount(conversation) == 0


def test_clear_removes_the_rows_rather_than_hiding_them(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    """A soft delete would leave a cleared conversation's history recoverable."""
    seed(store, conversation, 2)
    store.clear(conversation)

    rows = SQLService().query(
        select(ChatHistoryMessageDTO).where(
            ChatHistoryMessageDTO.conversation_id == conversation
        ),
        SQLResultFormats.all_scalar_list,
    )
    assert rows == []


def test_clearing_twice_is_not_an_error(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 1)
    store.clear(conversation)
    store.clear(conversation)
    assert store.count(conversation) == 0


def test_clearing_one_parent_leaves_the_others(
    store: SqlChatHistoryStore, conversation: ConversationId, other: ConversationId
) -> None:
    seed(store, conversation, 2)
    seed(store, other, 2)
    store.clear(conversation)
    assert store.count(other) == 2


# Replace range


def test_replace_range_swaps_the_items_up_to_the_boundary(
    store: SqlChatHistoryStore, conversation: ConversationId
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
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 4)
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


def test_replace_range_gives_the_replacements_their_ids(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 2)
    returned = store.replace_range(
        conversation, up_to_sequence=2, replacements=[message(conversation, 1, "s")]
    )
    assert returned[0].id is not None


def test_replace_range_keeps_everything_above_the_boundary(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 5)
    store.replace_range(
        conversation, up_to_sequence=3, replacements=[message(conversation, 3, "s")]
    )
    assert [item.sequence for item in store.get(conversation)] == [3, 4, 5]


def test_replace_range_on_a_parent_with_no_items_just_inserts(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    store.replace_range(
        conversation, up_to_sequence=1, replacements=[message(conversation, 1)]
    )
    assert store.count(conversation) == 1


def test_replace_range_leaves_another_parent_alone(
    store: SqlChatHistoryStore, conversation: ConversationId, other: ConversationId
) -> None:
    seed(store, conversation, 3)
    seed(store, other, 3)
    store.replace_range(
        conversation, up_to_sequence=3, replacements=[message(conversation, 1, "s")]
    )
    assert store.count(other) == 3


def test_replace_range_refuses_an_empty_replacement_list(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 2)
    with pytest.raises(ValueError, match="use clear\\(\\)"):
        store.replace_range(conversation, up_to_sequence=1, replacements=[])
    assert store.count(conversation) == 2


async def test_areplace_range_refuses_an_empty_replacement_list(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    with pytest.raises(ValueError, match="use clear\\(\\)"):
        await store.areplace_range(conversation, up_to_sequence=1, replacements=[])


def test_a_refused_replace_range_writes_nothing(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 2)
    with pytest.raises(ValueError):
        store.replace_range(
            conversation, up_to_sequence=1, replacements=[message(conversation, 9)]
        )
    assert [item.sequence for item in store.get(conversation)] == [1, 2]


# Mapping


def test_a_tool_call_survives_the_round_trip(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    call = StoredToolCall(
        id="call-1",
        function=StoredToolCallFunction(
            name="get_weather", arguments='{"city":"Oslo"}'
        ),
        provider_signature="sig",
    )
    store.append(
        conversation,
        [
            StoredMessage(
                conversation_id=conversation,
                sequence=1,
                role=MessageRole.ASSISTANT,
                tool_calls=[call],
            )
        ],
    )

    stored = store.get(conversation)[0]
    assert stored.tool_calls == [call]
    assert stored.content is None


def test_a_tool_reply_keeps_the_call_it_answers(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    store.append(
        conversation,
        [
            StoredMessage(
                conversation_id=conversation,
                sequence=1,
                role=MessageRole.TOOL,
                content="21 in Oslo",
                tool_call_id="call-1",
            )
        ],
    )

    stored = store.get(conversation)[0]
    assert (stored.role, stored.tool_call_id) == (MessageRole.TOOL, "call-1")


def test_a_message_with_no_tool_calls_reads_back_with_none_rather_than_an_empty_list(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 1)
    assert store.get(conversation)[0].tool_calls is None


def test_metadata_and_token_count_survive_the_round_trip(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    store.append(
        conversation,
        [
            StoredMessage(
                conversation_id=conversation,
                sequence=1,
                role=MessageRole.USER,
                content="hello",
                token_count=7,
                metadata={"is_policy_violated": False, "nested": {"a": 1}},
            )
        ],
    )

    stored = store.get(conversation)[0]
    assert stored.token_count == 7
    assert stored.metadata == {"is_policy_violated": False, "nested": {"a": 1}}


def test_absent_metadata_reads_back_as_an_empty_mapping(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 1)
    assert store.get(conversation)[0].metadata == {}


def test_the_parent_id_comes_back_as_a_conversation_id(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    seed(store, conversation, 1)
    assert store.get(conversation)[0].conversation_id == conversation


def test_awkward_unicode_survives_the_round_trip(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    content = "مرحبا 🙂\u200b café — \u0000 is not allowed, \\n is"
    store.append(conversation, [message(conversation, 1, content)])
    assert store.get(conversation)[0].content == content


def test_very_long_content_survives_the_round_trip(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    content = "x" * 200_000
    store.append(conversation, [message(conversation, 1, content)])
    assert store.get(conversation)[0].content == content


async def test_the_sync_and_async_paths_share_one_table(
    store: SqlChatHistoryStore, conversation: ConversationId
) -> None:
    """Two implementations over one table, not two stores."""
    store.append(conversation, [message(conversation, 1)])
    await store.aappend(conversation, [message(conversation, 2)])

    assert [item.sequence for item in await store.aget(conversation)] == [1, 2]
    assert [item.sequence for item in store.get(conversation)] == [1, 2]


# Identity


def test_the_store_is_one_instance_per_process(store: SqlChatHistoryStore) -> None:
    assert SqlChatHistoryStore() is store
