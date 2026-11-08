"""`stores/user_identity.py`: the link between a provider's subject and a user.

The lookup exists in both a sync and an async form, and they are separate
implementations rather than one wrapping the other, so each is held to the
same expectations. What the lookup must never do is match a link that was
deleted, or one belonging to a different provider.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from base.models.definitions.errors import DatabaseErrors
from base.tools.errors import ErrorsException
from identity.models.orm.user import UserDTO
from identity.models.orm.user_identity import UserIdentityDTO
from identity.stores.user import UserStore
from identity.stores.user_identity import UserIdentityStore


@pytest.fixture
def store(identity_settings: Any) -> UserIdentityStore:
    return UserIdentityStore()


@pytest.fixture
async def user(identity_settings: Any) -> UserDTO:
    return await UserStore().aregister(
        UserDTO(email="person@example.com", password_hash=None)
    )


async def link(
    store: UserIdentityStore,
    user: UserDTO,
    provider: str = "google",
    subject: str = "subject-1",
) -> UserIdentityDTO:
    return await store.aadd(
        UserIdentityDTO(
            user_id=user.id,
            provider=provider,
            subject=subject,
            email_at_link="person@example.com",
        )
    )


# Finding a link


async def test_a_link_is_found_by_its_provider_and_subject(
    store: UserIdentityStore, user: UserDTO
) -> None:
    await link(store, user)
    found = await store.aget_by_provider_subject("google", "subject-1")
    assert found is not None
    assert found.user_id == user.id


async def test_the_sync_lookup_finds_the_same_link(
    store: UserIdentityStore, user: UserDTO
) -> None:
    await link(store, user)
    found = store.get_by_provider_subject("google", "subject-1")
    assert found is not None
    assert found.user_id == user.id


async def test_a_subject_nobody_linked_is_not_found(
    store: UserIdentityStore, user: UserDTO
) -> None:
    await link(store, user)
    assert await store.aget_by_provider_subject("google", "stranger") is None


def test_the_sync_lookup_also_finds_nothing(store: UserIdentityStore) -> None:
    assert store.get_by_provider_subject("google", "stranger") is None


async def test_a_subject_at_another_provider_is_a_different_link(
    store: UserIdentityStore, user: UserDTO
) -> None:
    """Provider namespaces are separate; a subject only means something in one."""
    await link(store, user, provider="google")
    assert await store.aget_by_provider_subject("github", "subject-1") is None


async def test_the_sync_lookup_keeps_the_providers_apart(
    store: UserIdentityStore, user: UserDTO
) -> None:
    await link(store, user, provider="google")
    assert store.get_by_provider_subject("github", "subject-1") is None


# Deleted links


async def test_a_deleted_link_is_not_found(
    store: UserIdentityStore, user: UserDTO
) -> None:
    created = await link(store, user)
    await store.adelete(created.id)
    assert await store.aget_by_provider_subject("google", "subject-1") is None


async def test_the_sync_lookup_also_skips_a_deleted_link(
    store: UserIdentityStore, user: UserDTO
) -> None:
    created = await link(store, user)
    await store.adelete(created.id)
    assert store.get_by_provider_subject("google", "subject-1") is None


# What a link records, and what it may never be edited into


async def test_a_link_records_the_address_it_was_made_with(
    store: UserIdentityStore, user: UserDTO
) -> None:
    created = await link(store, user)
    assert created.email_at_link == "person@example.com"


async def test_an_edit_cannot_move_a_link_to_another_subject(
    store: UserIdentityStore, user: UserDTO
) -> None:
    """Rewriting a subject is rewriting which provider account owns a user."""
    created = await link(store, user)
    patch = UserIdentityDTO(
        user_id=user.id, provider="google", subject="someone-elses-subject"
    )
    await store.aedit(created.id, patch)
    assert (await store.aget(created.id)).subject == "subject-1"


async def test_an_edit_cannot_move_a_link_to_another_provider(
    store: UserIdentityStore, user: UserDTO
) -> None:
    created = await link(store, user)
    patch = UserIdentityDTO(user_id=user.id, provider="github", subject="subject-1")
    await store.aedit(created.id, patch)
    assert (await store.aget(created.id)).provider == "google"


def test_the_provider_and_subject_are_protected_on_top_of_the_defaults() -> None:
    assert {"provider", "subject"} <= UserIdentityStore.protected_fields


async def test_one_subject_cannot_be_linked_twice_at_a_provider(
    store: UserIdentityStore, user: UserDTO
) -> None:
    await link(store, user)
    with pytest.raises(ErrorsException) as caught:
        await link(store, user)
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


async def test_a_link_to_a_user_who_does_not_exist_is_not_found(
    store: UserIdentityStore,
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await store.aget(uuid.uuid4())
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND


def test_one_store_serves_the_whole_process(store: UserIdentityStore) -> None:
    assert UserIdentityStore() is store
