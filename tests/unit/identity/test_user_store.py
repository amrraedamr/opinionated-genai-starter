"""`stores/user.py`: a registration and the event announcing it, together.

The outbox is the point: the row and the event are one transaction, so a
service that crashed between them cannot have told anyone about a user it
never stored. Everything else here is the store's own policy — which columns
an edit may never reach, which ones a search matches.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import select

from base.models.definitions.errors import DatabaseErrors
from base.models.view_models.database import FilteredPaginationParams
from base.services.sql.service import SQLService
from base.tools.correlation import correlation_id
from base.tools.errors import ErrorsException
from contracts.v1.identity import USER_REGISTERED
from identity.models.orm.outbox import OutboxEventDTO
from identity.models.orm.user import UserDTO
from identity.stores.user import UserStore


@pytest.fixture
def store(identity_settings: Any) -> UserStore:
    return UserStore()


async def outbox_rows() -> list[OutboxEventDTO]:
    return await SQLService().aquery(
        select(OutboxEventDTO), lambda result: list(result.scalars().all())
    )


async def register(store: UserStore, email: str = "person@example.com") -> UserDTO:
    return await store.aregister(UserDTO(email=email, password_hash=None))


# Registering


async def test_registering_returns_the_stored_row(store: UserStore) -> None:
    user = await register(store)
    assert user.id is not None
    assert user.created_at is not None
    assert (await store.aget(user.id)).email == "person@example.com"


async def test_registering_announces_the_user(store: UserStore) -> None:
    user = await register(store)
    rows = await outbox_rows()
    assert len(rows) == 1
    assert rows[0].topic == USER_REGISTERED
    assert rows[0].event["payload"]["user_id"] == str(user.id)
    assert rows[0].event["payload"]["email"] == "person@example.com"


async def test_the_announcement_names_this_service_as_its_producer(
    store: UserStore,
) -> None:
    await register(store)
    rows = await outbox_rows()
    assert rows[0].event["meta"]["producer"] == "identity"


async def test_the_announcement_carries_the_request_that_caused_it(
    store: UserStore,
) -> None:
    """A user asking why they got the welcome email can be answered."""
    request = uuid.uuid4()
    correlation_id.set(request)
    await register(store)
    rows = await outbox_rows()
    assert rows[0].event["meta"]["correlation_id"] == str(request)


async def test_the_announcement_is_not_yet_published(store: UserStore) -> None:
    await register(store)
    rows = await outbox_rows()
    assert rows[0].published_at is None
    assert rows[0].attempts == 0


async def test_registering_an_address_twice_is_a_conflict(
    store: UserStore,
) -> None:
    await register(store)
    with pytest.raises(ErrorsException) as caught:
        await register(store)
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


async def test_a_refused_registration_announces_nothing(store: UserStore) -> None:
    """Row and event commit together, so a rejected write leaves no event."""
    await register(store)
    with pytest.raises(ErrorsException):
        await register(store)
    assert len(await outbox_rows()) == 1


async def test_two_users_each_get_their_own_announcement(
    store: UserStore,
) -> None:
    await register(store, "first@example.com")
    await register(store, "second@example.com")
    rows = await outbox_rows()
    assert len({row.id for row in rows}) == 2


# What an edit may not touch


async def test_an_edit_cannot_change_the_address(store: UserStore) -> None:
    user = await register(store)
    patch = UserDTO(email="taken-over@example.com", password_hash=None)
    await store.aedit(user.id, patch)
    assert (await store.aget(user.id)).email == "person@example.com"


def test_the_address_is_protected_on_top_of_the_defaults() -> None:
    assert "email" in UserStore.protected_fields
    assert {"id", "created_at", "updated_at", "deleted_at"} <= (
        UserStore.protected_fields
    )


async def test_an_edit_can_change_a_name(store: UserStore) -> None:
    user = await register(store)
    patch = UserDTO(email="ignored@example.com", password_hash=None, first_name="Ada")
    await store.aedit(user.id, patch)
    assert (await store.aget(user.id)).first_name == "Ada"


def test_an_edit_to_a_user_who_does_not_exist_returns_nothing(
    store: UserStore,
) -> None:
    patch = UserDTO(email="x@example.com", password_hash=None)
    assert store.edit(uuid.uuid4(), patch) is None


# Searching


async def test_a_search_matches_part_of_an_address(store: UserStore) -> None:
    await register(store, "ada.lovelace@example.com")
    await register(store, "grace.hopper@example.com")
    page = await store.aget_all_paged(FilteredPaginationParams(search="lovelace"))
    assert [row.email for row in page.data] == ["ada.lovelace@example.com"]


def test_the_address_is_the_only_column_a_search_matches() -> None:
    assert UserStore.search_fields == ("email",)


# Soft deletion


async def test_a_deleted_user_is_no_longer_found(store: UserStore) -> None:
    user = await register(store)
    assert await store.adelete(user.id) is True
    with pytest.raises(ErrorsException) as caught:
        await store.aget(user.id)
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND


async def test_deleting_a_user_twice_reports_the_second_as_nothing_to_do(
    store: UserStore,
) -> None:
    user = await register(store)
    await store.adelete(user.id)
    assert await store.adelete(user.id) is False


async def test_a_deleted_address_is_still_taken(store: UserStore) -> None:
    """A soft delete leaves the row, and the unique index with it."""
    user = await register(store)
    await store.adelete(user.id)
    with pytest.raises(ErrorsException) as caught:
        await register(store)
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


# The store itself


def test_one_store_serves_the_whole_process(store: UserStore) -> None:
    assert UserStore() is store
