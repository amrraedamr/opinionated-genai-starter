"""`ConversationStore`, whose whole job is that ownership is part of the query.

Every method here takes the owner as its first argument rather than checking
one afterwards, so the tests that matter most are the ones where a second user
asks for the first user's row. Sync and async are separate implementations, so
both are exercised against the same expectations, against SQLite.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import ConversationId, StoredMessage
from base.models.view_models.database import FilteredPaginationParams
from base.services.sql.service import SQLService
from base.tools.database import SQLResultFormats
from chat_api.models.orm.chat_history_message import ChatHistoryMessageDTO
from chat_api.models.orm.conversation import ConversationDTO
from chat_api.models.orm.outbox import OutboxEventDTO
from chat_api.models.view_models.conversation import (
    CreateConversationRequest,
    EditConversationRequest,
)
from chat_api.stores.chat_history.sql import SqlChatHistoryStore
from chat_api.stores.conversation import ConversationStore
from contracts.v1.chat import CONVERSATION_CREATED


@pytest.fixture
def store(tables: None) -> ConversationStore:
    return ConversationStore()


@pytest.fixture
def mine(store: ConversationStore, user_id: uuid.UUID) -> ConversationDTO:
    return store.create_for_user(user_id, CreateConversationRequest(title="mine"))


@pytest.fixture
def theirs(store: ConversationStore, other_user_id: uuid.UUID) -> ConversationDTO:
    return store.create_for_user(
        other_user_id, CreateConversationRequest(title="theirs")
    )


def outbox_events() -> list[OutboxEventDTO]:
    return SQLService().query(select(OutboxEventDTO), SQLResultFormats.all_scalar_list)


# Create


def test_create_stores_the_title_and_metadata_it_was_given(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    created = store.create_for_user(
        user_id, CreateConversationRequest(title="A", metadata={"k": "v"})
    )

    assert (created.title, created.meta, created.user_id) == ("A", {"k": "v"}, user_id)
    assert created.id is not None


async def test_acreate_stores_the_title_and_metadata_it_was_given(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    created = await store.acreate_for_user(
        user_id, CreateConversationRequest(title="A", metadata={"k": "v"})
    )

    assert (created.title, created.meta, created.user_id) == ("A", {"k": "v"}, user_id)


async def test_acreate_stages_the_event_in_the_same_transaction_as_the_row(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    created = await store.acreate_for_user(user_id, CreateConversationRequest())

    events = outbox_events()
    assert [event.topic for event in events] == [CONVERSATION_CREATED]
    assert events[0].event["payload"] == {
        "conversation_id": str(created.id),
        "user_id": str(user_id),
        "created_at": created.created_at.isoformat().replace("+00:00", "Z"),
    }
    assert events[0].published_at is None


async def test_the_event_names_this_service_as_its_producer(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    await store.acreate_for_user(user_id, CreateConversationRequest())
    assert outbox_events()[0].event["meta"]["producer"] == "chat_api"


def test_only_the_async_create_publishes_an_event(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    """Pinned asymmetry: the sync path writes the row and nothing else."""
    store.create_for_user(user_id, CreateConversationRequest())
    assert outbox_events() == []


# Get


def test_get_finds_the_callers_own_conversation(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    assert store.get_for_user(user_id, mine.id).id == mine.id


async def test_aget_finds_the_callers_own_conversation(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    found = await store.aget_for_user(user_id, mine.id)
    assert found is not None and found.id == mine.id


def test_get_refuses_a_conversation_owned_by_somebody_else(
    store: ConversationStore, user_id: uuid.UUID, theirs: ConversationDTO
) -> None:
    assert store.get_for_user(user_id, theirs.id) is None


async def test_aget_refuses_a_conversation_owned_by_somebody_else(
    store: ConversationStore, user_id: uuid.UUID, theirs: ConversationDTO
) -> None:
    assert await store.aget_for_user(user_id, theirs.id) is None


def test_get_of_an_id_that_was_never_stored_is_none(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    assert store.get_for_user(user_id, uuid.uuid4()) is None


def test_a_deleted_conversation_is_no_longer_found(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    store.delete_for_user(user_id, mine.id)
    assert store.get_for_user(user_id, mine.id) is None


# Get many


def test_get_many_returns_only_the_callers_rows(
    store: ConversationStore,
    user_id: uuid.UUID,
    mine: ConversationDTO,
    theirs: ConversationDTO,
) -> None:
    rows = store.get_many_for_user(user_id, [mine.id, theirs.id, uuid.uuid4()])
    assert [row.id for row in rows] == [mine.id]


async def test_aget_many_returns_only_the_callers_rows(
    store: ConversationStore,
    user_id: uuid.UUID,
    mine: ConversationDTO,
    theirs: ConversationDTO,
) -> None:
    rows = await store.aget_many_for_user(user_id, [mine.id, theirs.id])
    assert [row.id for row in rows] == [mine.id]


def test_get_many_of_nothing_is_empty(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    assert store.get_many_for_user(user_id, []) == []


async def test_aget_many_of_nothing_is_empty(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    assert await store.aget_many_for_user(user_id, []) == []


# Get all


def test_get_all_returns_every_conversation_the_caller_owns(
    store: ConversationStore,
    user_id: uuid.UUID,
    mine: ConversationDTO,
    theirs: ConversationDTO,
) -> None:
    store.create_for_user(user_id, CreateConversationRequest(title="second"))
    assert {row.title for row in store.get_all_for_user(user_id)} == {
        "mine",
        "second",
    }


async def test_aget_all_returns_every_conversation_the_caller_owns(
    store: ConversationStore,
    user_id: uuid.UUID,
    mine: ConversationDTO,
    theirs: ConversationDTO,
) -> None:
    rows = await store.aget_all_for_user(user_id)
    assert [row.id for row in rows] == [mine.id]


def test_get_all_for_a_user_with_nothing_is_empty(
    store: ConversationStore, theirs: ConversationDTO
) -> None:
    assert store.get_all_for_user(uuid.uuid4()) == []


def test_get_all_leaves_out_what_was_deleted(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    store.delete_for_user(user_id, mine.id)
    assert store.get_all_for_user(user_id) == []


# Pagination


@pytest.fixture
def seven(store: ConversationStore, user_id: uuid.UUID) -> list[ConversationDTO]:
    return [
        store.create_for_user(user_id, CreateConversationRequest(title=f"chat-{i}"))
        for i in range(7)
    ]


def page(store: ConversationStore, user_id: uuid.UUID, **params: Any) -> Any:
    return store.get_all_paged_for_user(user_id, FilteredPaginationParams(**params))


@pytest.mark.parametrize(
    ("number", "size", "titles"),
    [
        (1, 3, ["chat-0", "chat-1", "chat-2"]),
        (2, 3, ["chat-3", "chat-4", "chat-5"]),
        (3, 3, ["chat-6"]),
        (1, 100, [f"chat-{i}" for i in range(7)]),
    ],
)
def test_a_page_holds_the_rows_its_offset_points_at(
    store: ConversationStore,
    user_id: uuid.UUID,
    seven: list[ConversationDTO],
    number: int,
    size: int,
    titles: list[str],
) -> None:
    result = page(store, user_id, page=number, size=size)
    assert [row.title for row in result.data] == titles


def test_a_page_reports_the_whole_total_not_the_page(
    store: ConversationStore, user_id: uuid.UUID, seven: list[ConversationDTO]
) -> None:
    result = page(store, user_id, page=1, size=2)
    assert (len(result.data), result.total, result.total_pages) == (2, 7, 4)


def test_a_page_past_the_end_is_empty_and_still_reports_the_total(
    store: ConversationStore, user_id: uuid.UUID, seven: list[ConversationDTO]
) -> None:
    result = page(store, user_id, page=99, size=3)
    assert (result.data, result.total) == ([], 7)


async def test_an_async_page_past_the_end_is_empty_and_still_reports_the_total(
    store: ConversationStore, user_id: uuid.UUID, seven: list[ConversationDTO]
) -> None:
    result = await store.aget_all_paged_for_user(
        user_id, FilteredPaginationParams(page=99, size=3)
    )
    assert (result.data, result.total) == ([], 7)


def test_a_page_counts_only_the_callers_rows(
    store: ConversationStore,
    user_id: uuid.UUID,
    seven: list[ConversationDTO],
    theirs: ConversationDTO,
) -> None:
    result = page(store, user_id, page=1, size=100)
    assert result.total == 7
    assert all(row.user_id == user_id for row in result.data)


async def test_an_async_page_counts_only_the_callers_rows(
    store: ConversationStore,
    other_user_id: uuid.UUID,
    seven: list[ConversationDTO],
    theirs: ConversationDTO,
) -> None:
    result = await store.aget_all_paged_for_user(
        other_user_id, FilteredPaginationParams(page=1, size=100)
    )
    assert [row.id for row in result.data] == [theirs.id]


def test_search_matches_the_title_and_nothing_else(
    store: ConversationStore, user_id: uuid.UUID, seven: list[ConversationDTO]
) -> None:
    result = page(store, user_id, page=1, size=100, search="chat-3")
    assert [row.title for row in result.data] == ["chat-3"]


def test_search_is_case_insensitive(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    store.create_for_user(user_id, CreateConversationRequest(title="Quarterly Review"))
    result = page(store, user_id, page=1, size=100, search="quarterly")
    assert len(result.data) == 1


def test_a_search_that_matches_nothing_reports_no_pages(
    store: ConversationStore, user_id: uuid.UUID, seven: list[ConversationDTO]
) -> None:
    result = page(store, user_id, page=1, size=100, search="nothing like this")
    assert (result.data, result.total, result.total_pages) == ([], 0, 0)


def test_a_search_cannot_reach_another_users_rows(
    store: ConversationStore, user_id: uuid.UUID, theirs: ConversationDTO
) -> None:
    assert page(store, user_id, page=1, size=100, search="theirs").data == []


def test_an_empty_table_reports_no_pages(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    result = page(store, user_id, page=1, size=10)
    assert (result.data, result.total, result.total_pages) == ([], 0, 0)


def test_the_page_size_field_is_named_size(
    store: ConversationStore, user_id: uuid.UUID, seven: list[ConversationDTO]
) -> None:
    """A misspelled page size is refused rather than quietly ignored."""
    with pytest.raises(ValidationError):
        FilteredPaginationParams(page_size=2)


# Add


def test_add_stamps_the_owner_over_whatever_the_entity_claimed(
    store: ConversationStore, user_id: uuid.UUID, other_user_id: uuid.UUID
) -> None:
    added = store.add_for_user(
        user_id, ConversationDTO(user_id=other_user_id, title="t")
    )
    assert added.user_id == user_id


async def test_aadd_stamps_the_owner_over_whatever_the_entity_claimed(
    store: ConversationStore, user_id: uuid.UUID, other_user_id: uuid.UUID
) -> None:
    added = await store.aadd_for_user(
        user_id, ConversationDTO(user_id=other_user_id, title="t")
    )
    assert added.user_id == user_id


def test_add_many_stamps_the_owner_on_every_row(
    store: ConversationStore, user_id: uuid.UUID, other_user_id: uuid.UUID
) -> None:
    rows = store.add_many_for_user(
        user_id,
        [ConversationDTO(user_id=other_user_id, title=t) for t in ("a", "b")],
    )
    assert [row.user_id for row in rows] == [user_id, user_id]


async def test_aadd_many_stamps_the_owner_on_every_row(
    store: ConversationStore, user_id: uuid.UUID, other_user_id: uuid.UUID
) -> None:
    rows = await store.aadd_many_for_user(
        user_id,
        [ConversationDTO(user_id=other_user_id, title=t) for t in ("a", "b")],
    )
    assert [row.user_id for row in rows] == [user_id, user_id]


def test_add_many_of_nothing_writes_nothing(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    assert store.add_many_for_user(user_id, []) == []
    assert store.get_all_for_user(user_id) == []


# Edit


def test_edit_writes_the_title(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    edited = store.edit_for_user(
        user_id, mine.id, EditConversationRequest(title="renamed")
    )
    assert edited is not None and edited.title == "renamed"


async def test_aedit_writes_the_title(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    edited = await store.aedit_for_user(
        user_id, mine.id, EditConversationRequest(title="renamed")
    )
    assert edited is not None and edited.title == "renamed"


def test_edit_renames_metadata_onto_the_column_that_holds_it(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    """The request calls it `metadata`; the table calls it `meta`."""
    edited = store.edit_for_user(
        user_id, mine.id, EditConversationRequest(metadata={"k": "v"})
    )
    assert edited is not None and edited.meta == {"k": "v"}


async def test_aedit_renames_metadata_onto_the_column_that_holds_it(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    edited = await store.aedit_for_user(
        user_id, mine.id, EditConversationRequest(metadata={"k": "v"})
    )
    assert edited is not None and edited.meta == {"k": "v"}


def test_a_field_the_caller_left_out_is_left_alone(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    created = store.create_for_user(
        user_id, CreateConversationRequest(title="t", metadata={"k": "v"})
    )
    store.edit_for_user(user_id, created.id, EditConversationRequest(title="renamed"))

    reloaded = store.get_for_user(user_id, created.id)
    assert reloaded is not None and reloaded.meta == {"k": "v"}


def test_an_explicit_none_is_not_taken_as_a_change(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    store.edit_for_user(user_id, mine.id, EditConversationRequest(title=None))

    reloaded = store.get_for_user(user_id, mine.id)
    assert reloaded is not None and reloaded.title == "mine"


def test_edit_cannot_reach_a_protected_column(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    """`user_id` is protected, so a patch carrying one cannot move ownership."""
    request = EditConversationRequest(title="t")
    object.__setattr__(request, "user_id", uuid.uuid4())
    store.edit_for_user(user_id, mine.id, request)

    reloaded = store.get_for_user(user_id, mine.id)
    assert reloaded is not None and reloaded.user_id == user_id


def test_edit_of_a_conversation_that_does_not_exist_is_none(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    assert (
        store.edit_for_user(user_id, uuid.uuid4(), EditConversationRequest(title="x"))
        is None
    )


async def test_aedit_of_a_conversation_that_does_not_exist_is_none(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    assert (
        await store.aedit_for_user(
            user_id, uuid.uuid4(), EditConversationRequest(title="x")
        )
        is None
    )


def test_edit_refuses_a_conversation_owned_by_somebody_else(
    store: ConversationStore, user_id: uuid.UUID, theirs: ConversationDTO
) -> None:
    assert (
        store.edit_for_user(user_id, theirs.id, EditConversationRequest(title="x"))
        is None
    )
    assert store.get_for_user(theirs.user_id, theirs.id).title == "theirs"


async def test_aedit_refuses_a_conversation_owned_by_somebody_else(
    store: ConversationStore, user_id: uuid.UUID, theirs: ConversationDTO
) -> None:
    assert (
        await store.aedit_for_user(
            user_id, theirs.id, EditConversationRequest(title="x")
        )
        is None
    )


def test_a_title_may_be_awkward_unicode(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    title = "مرحبا 🙂\u200b café"
    store.edit_for_user(user_id, mine.id, EditConversationRequest(title=title))

    reloaded = store.get_for_user(user_id, mine.id)
    assert reloaded is not None and reloaded.title == title


# Delete


def test_delete_soft_deletes_rather_than_removing(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    assert store.delete_for_user(user_id, mine.id) is True

    row = SQLService().query(
        select(ConversationDTO).where(ConversationDTO.id == mine.id),
        SQLResultFormats.one_or_none_scalar,
    )
    assert row is not None and row.deleted_at is not None


async def test_adelete_soft_deletes_rather_than_removing(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    assert await store.adelete_for_user(user_id, mine.id) is True
    assert await store.aget_for_user(user_id, mine.id) is None


def test_delete_of_a_conversation_that_does_not_exist_is_false(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    assert store.delete_for_user(user_id, uuid.uuid4()) is False


async def test_adelete_of_a_conversation_that_does_not_exist_is_false(
    store: ConversationStore, user_id: uuid.UUID
) -> None:
    assert await store.adelete_for_user(user_id, uuid.uuid4()) is False


def test_delete_refuses_a_conversation_owned_by_somebody_else(
    store: ConversationStore, user_id: uuid.UUID, theirs: ConversationDTO
) -> None:
    assert store.delete_for_user(user_id, theirs.id) is False
    assert store.get_for_user(theirs.user_id, theirs.id) is not None


async def test_adelete_refuses_a_conversation_owned_by_somebody_else(
    store: ConversationStore, user_id: uuid.UUID, theirs: ConversationDTO
) -> None:
    assert await store.adelete_for_user(user_id, theirs.id) is False


def test_deleting_twice_is_false_the_second_time(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    store.delete_for_user(user_id, mine.id)
    assert store.delete_for_user(user_id, mine.id) is False


def seed_messages(conversation_id: uuid.UUID, count: int = 2) -> None:
    SqlChatHistoryStore().append(
        ConversationId(conversation_id),
        [
            StoredMessage(
                conversation_id=ConversationId(conversation_id),
                sequence=i,
                role=MessageRole.USER,
                content=f"m{i}",
            )
            for i in range(1, count + 1)
        ],
    )


def live_messages(conversation_id: uuid.UUID) -> list[ChatHistoryMessageDTO]:
    return SQLService().query(
        select(ChatHistoryMessageDTO).where(
            ChatHistoryMessageDTO.conversation_id == conversation_id,
            ChatHistoryMessageDTO.deleted_at.is_(None),
        ),
        SQLResultFormats.all_scalar_list,
    )


def test_deleting_a_conversation_cascades_to_its_messages(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    seed_messages(mine.id)
    store.delete_for_user(user_id, mine.id)
    assert live_messages(mine.id) == []


async def test_an_async_delete_cascades_to_its_messages(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    seed_messages(mine.id)
    await store.adelete_for_user(user_id, mine.id)
    assert live_messages(mine.id) == []


def test_a_cascade_leaves_another_conversations_messages_alone(
    store: ConversationStore, user_id: uuid.UUID, mine: ConversationDTO
) -> None:
    other = store.create_for_user(user_id, CreateConversationRequest(title="other"))
    seed_messages(mine.id)
    seed_messages(other.id)

    store.delete_for_user(user_id, mine.id)
    assert len(live_messages(other.id)) == 2


# Identity


def test_the_store_is_one_instance_per_process(store: ConversationStore) -> None:
    assert ConversationStore() is store


def test_the_owner_column_is_protected_from_edits() -> None:
    assert "user_id" in ConversationStore.protected_fields
    assert ConversationStore.search_fields == ("title",)
