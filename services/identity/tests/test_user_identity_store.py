"""UserIdentityStore looks a social account up by the provider's own identifiers."""

from typing import Any
from uuid import uuid4

import pytest


@pytest.fixture
def store(configured_settings: None) -> Any:
    from identity.stores.user_identity import UserIdentityStore

    return UserIdentityStore()


def _identity(provider: str, subject: str) -> Any:
    from identity.models.orm.user_identity import UserIdentityDTO

    return UserIdentityDTO(user_id=uuid4(), provider=provider, subject=subject)


class TestProviderSubjectLookup:
    def test_it_finds_the_matching_identity(self, store):
        subject = uuid4().hex
        stored = store.add(_identity("google", subject))

        found = store.get_by_provider_subject("google", subject)
        assert found is not None
        assert found.id == stored.id

    def test_it_does_not_match_another_provider(self, store):
        subject = uuid4().hex
        store.add(_identity("google", subject))

        assert store.get_by_provider_subject("github", subject) is None

    def test_a_deleted_identity_is_not_found(self, store):
        subject = uuid4().hex
        stored = store.add(_identity("google", subject))
        store.delete(stored.id)

        assert store.get_by_provider_subject("google", subject) is None


class TestProtectedFields:
    def test_edit_cannot_change_the_provider_identifiers(self, store):
        subject = uuid4().hex
        stored = store.add(_identity("google", subject))

        updated = store.edit(stored.id, _identity("github", "other"))

        assert updated is not None
        assert updated.provider == "google"
        assert updated.subject == subject
