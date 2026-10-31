"""UserStore held to the contract every AbstractStore implementation shares."""

from __future__ import annotations

import uuid
from typing import Any

from base.testing import AbstractStoreConformance


class TestUserStoreConformance(AbstractStoreConformance):
    def make_store(self) -> Any:
        from base.services.store.user import UserStore

        return UserStore()

    def make_entity(self, **overrides: Any) -> Any:
        from base.models.orm.user import UserDTO

        return UserDTO(
            email=overrides.get("email", f"{uuid.uuid4().hex}@example.com"),
            password_hash="hash",
            first_name="Test",
            last_name="User",
        )


class TestConversationStore:
    def test_a_conversation_is_scoped_to_its_owner(self):
        from base.models.orm.conversation import ConversationDTO
        from base.services.store.conversation import ConversationStore

        store = ConversationStore()
        owner, stranger = uuid.uuid4(), uuid.uuid4()
        conversation = store.add_for_user(
            owner, ConversationDTO(user_id=owner, title="mine")
        )

        assert store.get_for_user(owner, conversation.id) is not None
        assert store.get_for_user(stranger, conversation.id) is None

    def test_deleting_a_conversation_takes_its_messages(self):
        from base.models.definitions.message import MessageRole
        from base.models.orm.chat_history_message import ChatHistoryMessageDTO
        from base.models.orm.conversation import ConversationDTO
        from base.services.store.conversation import ConversationStore

        store = ConversationStore()
        owner = uuid.uuid4()
        conversation = store.add_for_user(
            owner, ConversationDTO(user_id=owner, title="doomed")
        )
        store._sql_service.upsert_rows(
            [
                ChatHistoryMessageDTO(
                    conversation_id=conversation.id,
                    sequence=1,
                    role=MessageRole.USER,
                    content="hi",
                )
            ]
        )

        assert store._cascade(conversation) != []
        assert store.delete_for_user(owner, conversation.id) is True
        assert store._cascade(conversation) == []

    def test_edit_cannot_reassign_the_owner(self):
        from base.models.orm.conversation import ConversationDTO
        from base.services.store.conversation import ConversationStore

        store = ConversationStore()
        owner, stranger = uuid.uuid4(), uuid.uuid4()
        conversation = store.add_for_user(
            owner, ConversationDTO(user_id=owner, title="before")
        )

        replacement = ConversationDTO(user_id=stranger, title="after")
        updated = store.edit(conversation.id, replacement)

        assert updated is not None
        assert updated.user_id == owner
        assert updated.title == "after"


class TestUserIdentityStore:
    def test_it_finds_an_account_by_provider_and_subject(self):
        from base.models.orm.user_identity import UserIdentityDTO
        from base.services.store.user_identity import UserIdentityStore

        store = UserIdentityStore()
        subject = uuid.uuid4().hex
        stored = store.add(
            UserIdentityDTO(user_id=uuid.uuid4(), provider="google", subject=subject)
        )

        found = store.get_by_provider_subject("google", subject)
        assert found is not None and found.id == stored.id
        assert store.get_by_provider_subject("github", subject) is None

    def test_edit_cannot_change_the_provider_identifiers(self):
        from base.models.orm.user_identity import UserIdentityDTO
        from base.services.store.user_identity import UserIdentityStore

        store = UserIdentityStore()
        subject = uuid.uuid4().hex
        stored = store.add(
            UserIdentityDTO(user_id=uuid.uuid4(), provider="google", subject=subject)
        )

        updated = store.edit(
            stored.id,
            UserIdentityDTO(user_id=uuid.uuid4(), provider="github", subject="other"),
        )

        assert updated is not None
        assert updated.provider == "google"
        assert updated.subject == subject
