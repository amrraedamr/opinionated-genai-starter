"""ConversationStore keeps ownership in the query and takes messages down with it."""

from typing import Any
from uuid import uuid4

import pytest

from base.models.view_models.database import FilteredPaginationParams


@pytest.fixture
def store(configured_settings: None) -> Any:
    from chat_api.stores.conversation import ConversationStore

    return ConversationStore()


def _message(conversation_id: Any, sequence: int) -> Any:
    from base.models.definitions.message import MessageRole
    from chat_api.models.orm.chat_history_message import ChatHistoryMessageDTO

    return ChatHistoryMessageDTO(
        conversation_id=conversation_id,
        sequence=sequence,
        role=MessageRole.USER,
        content=f"message {sequence}",
        meta={},
    )


class TestOwnership:
    def test_another_user_cannot_read_the_conversation(self, store):
        owner, stranger = uuid4(), uuid4()
        conversation = store.add_for_user(owner, _conversation())

        assert store.get_for_user(owner, conversation.id) is not None
        assert store.get_for_user(stranger, conversation.id) is None

    def test_another_user_cannot_delete_the_conversation(self, store):
        owner, stranger = uuid4(), uuid4()
        conversation = store.add_for_user(owner, _conversation())

        assert store.delete_for_user(stranger, conversation.id) is False
        assert store.get_for_user(owner, conversation.id) is not None

    def test_get_all_for_user_excludes_other_owners(self, store):
        owner, stranger = uuid4(), uuid4()
        store.add_for_user(owner, _conversation())
        store.add_for_user(stranger, _conversation())

        assert len(store.get_all_for_user(owner)) == 1


class TestCascade:
    def test_delete_soft_deletes_the_messages_too(self, store):
        owner = uuid4()
        conversation = store.add_for_user(owner, _conversation())
        store._sql_service.upsert_rows(
            [_message(conversation.id, 1), _message(conversation.id, 2)]
        )

        assert store._cascade(conversation) != []
        assert store.delete_for_user(owner, conversation.id) is True

        survivors = store._cascade(conversation)
        assert survivors == []


class TestEdit:
    def test_edit_cannot_reassign_the_owner(self, store):
        owner, stranger = uuid4(), uuid4()
        conversation = store.add_for_user(owner, _conversation("before"))

        replacement = _conversation("after")
        replacement.user_id = stranger
        updated = store.edit(conversation.id, replacement)

        assert updated is not None
        assert updated.user_id == owner
        assert updated.title == "after"


class TestSearch:
    def test_paged_search_matches_the_title(self, store):
        owner = uuid4()
        store.add_for_user(owner, _conversation("findable penguin"))
        store.add_for_user(owner, _conversation("unrelated"))

        page = store.get_all_paged_for_user(
            owner, FilteredPaginationParams(page=1, size=10, search="penguin")
        )
        assert page.total == 1


def _conversation(title: str = "title") -> Any:
    from chat_api.models.orm.conversation import ConversationDTO

    return ConversationDTO(user_id=uuid4(), title=title, meta={})
