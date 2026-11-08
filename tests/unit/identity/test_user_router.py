"""`routers/user.py`: the account a caller may act on is their own.

Every route here resolves the user from the token rather than from anything
the caller sends, so the ownership tests are about what a second user's token
cannot reach. The password change is the one write with a precondition, and
it has three ways to be refused.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from base.tools.security import PasswordHashingService
from identity.models.orm.user import UserDTO
from identity.routers.auth import router as auth_router
from identity.routers.user import router as user_router
from identity.services.auth import AuthService
from identity.stores.user import UserStore

from .conftest import client_for

PASSWORD = "Str0ng!pass"
NEW_PASSWORD = "Ev3n!stronger"


@pytest.fixture
def client(identity_settings: Any) -> Iterator[TestClient]:
    with client_for(auth_router, user_router) as test_client:
        yield test_client


async def account(
    email: str = "person@example.com", password: str | None = PASSWORD
) -> tuple[UserDTO, dict[str, str]]:
    """A registered user and a bearer header for them."""
    user = await UserStore().aregister(
        UserDTO(
            email=email,
            password_hash=PasswordHashingService.hash(password) if password else None,
            first_name="Ada",
            last_name="Lovelace",
        )
    )
    pair = AuthService().generate_token_pair(user)
    return user, {"Authorization": f"Bearer {pair.access_token}"}


# Reading your own account


async def test_a_caller_reads_the_account_their_token_names(
    client: TestClient,
) -> None:
    user, headers = await account()
    body = client.get("/users/me", headers=headers).json()["data"]
    assert body["userId"] == str(user.id)
    assert body["email"] == "person@example.com"
    assert body["firstName"] == "Ada"


async def test_a_caller_never_reads_another_account(client: TestClient) -> None:
    await account("first@example.com")
    _, second_headers = await account("second@example.com")
    body = client.get("/users/me", headers=second_headers).json()["data"]
    assert body["email"] == "second@example.com"


def test_a_request_with_no_token_is_turned_away(client: TestClient) -> None:
    assert client.get("/users/me").status_code == 401


def test_a_token_this_service_never_signed_is_unauthorized(
    client: TestClient,
) -> None:
    response = client.get("/users/me", headers={"Authorization": "Bearer nonsense"})
    assert response.status_code == 401


async def test_a_refresh_token_is_not_a_credential(client: TestClient) -> None:
    """It buys a new pair at `/auth/refresh` and nothing else."""
    user, _ = await account()
    pair = AuthService().generate_token_pair(user)
    response = client.get(
        "/users/me", headers={"Authorization": f"Bearer {pair.refresh_token}"}
    )
    assert response.status_code == 401


async def test_a_token_for_a_deleted_account_is_a_record_not_found(
    client: TestClient,
) -> None:
    user, headers = await account()
    await UserStore().adelete(user.id)
    assert client.get("/users/me", headers=headers).status_code == 404


# Editing your own account


async def test_a_name_change_is_kept(client: TestClient) -> None:
    user, headers = await account()
    response = client.patch("/users/me", headers=headers, json={"firstName": "Grace"})
    assert response.status_code == 200
    assert (await UserStore().aget(user.id)).first_name == "Grace"


async def test_a_field_the_request_omits_is_left_alone(client: TestClient) -> None:
    user, headers = await account()
    client.patch("/users/me", headers=headers, json={"firstName": "Grace"})
    assert (await UserStore().aget(user.id)).last_name == "Lovelace"


async def test_an_address_sent_to_the_edit_route_is_ignored(
    client: TestClient,
) -> None:
    """The address is the account's identity; changing it is not an edit."""
    user, headers = await account()
    client.patch("/users/me", headers=headers, json={"email": "new@example.com"})
    assert (await UserStore().aget(user.id)).email == "person@example.com"


async def test_a_name_with_digits_in_it_is_refused(client: TestClient) -> None:
    _, headers = await account()
    response = client.patch("/users/me", headers=headers, json={"firstName": "Ada2"})
    assert response.status_code == 422


def test_an_edit_with_no_token_is_turned_away(client: TestClient) -> None:
    assert client.patch("/users/me", json={"firstName": "Grace"}).status_code == 401


# Changing your password


async def test_a_password_change_takes_effect(client: TestClient) -> None:
    _, headers = await account()
    response = client.patch(
        "/users/me",
        headers=headers,
        json={"currentPassword": PASSWORD, "newPassword": NEW_PASSWORD},
    )
    assert response.status_code == 200
    login = client.post(
        "/auth/login", json={"email": "person@example.com", "password": NEW_PASSWORD}
    )
    assert login.status_code == 200


async def test_the_old_password_stops_working(client: TestClient) -> None:
    _, headers = await account()
    client.patch(
        "/users/me",
        headers=headers,
        json={"currentPassword": PASSWORD, "newPassword": NEW_PASSWORD},
    )
    login = client.post(
        "/auth/login", json={"email": "person@example.com", "password": PASSWORD}
    )
    assert login.status_code == 401


async def test_the_new_password_is_not_stored_as_written(
    client: TestClient,
) -> None:
    user, headers = await account()
    client.patch(
        "/users/me",
        headers=headers,
        json={"currentPassword": PASSWORD, "newPassword": NEW_PASSWORD},
    )
    stored = await UserStore().aget(user.id)
    assert stored.password_hash is not None
    assert NEW_PASSWORD not in stored.password_hash


async def test_a_password_change_proving_nothing_is_unauthorized(
    client: TestClient,
) -> None:
    """The stolen-token case: holding one must not be enough to reset it."""
    _, headers = await account()
    response = client.patch(
        "/users/me",
        headers=headers,
        json={"currentPassword": "Wr0ng!pass", "newPassword": NEW_PASSWORD},
    )
    assert response.status_code == 401


async def test_an_account_with_no_password_cannot_change_one(
    client: TestClient,
) -> None:
    _, headers = await account("oauth@example.com", password=None)
    response = client.patch(
        "/users/me",
        headers=headers,
        json={"currentPassword": PASSWORD, "newPassword": NEW_PASSWORD},
    )
    assert response.status_code == 400


async def test_a_new_password_with_no_current_one_is_refused(
    client: TestClient,
) -> None:
    _, headers = await account()
    response = client.patch(
        "/users/me", headers=headers, json={"newPassword": NEW_PASSWORD}
    )
    assert response.status_code == 422


async def test_a_current_password_with_no_new_one_is_refused(
    client: TestClient,
) -> None:
    _, headers = await account()
    response = client.patch(
        "/users/me", headers=headers, json={"currentPassword": PASSWORD}
    )
    assert response.status_code == 422


async def test_a_new_password_too_weak_to_accept_is_refused(
    client: TestClient,
) -> None:
    _, headers = await account()
    response = client.patch(
        "/users/me",
        headers=headers,
        json={"currentPassword": PASSWORD, "newPassword": "short"},
    )
    assert response.status_code == 422


async def test_a_failed_password_change_leaves_the_old_one_working(
    client: TestClient,
) -> None:
    _, headers = await account()
    client.patch(
        "/users/me",
        headers=headers,
        json={"currentPassword": "Wr0ng!pass", "newPassword": NEW_PASSWORD},
    )
    login = client.post(
        "/auth/login", json={"email": "person@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200


# Deleting your own account


async def test_deleting_an_account_succeeds(client: TestClient) -> None:
    _, headers = await account()
    assert client.delete("/users/me", headers=headers).status_code == 200


async def test_a_deleted_account_can_no_longer_be_read(client: TestClient) -> None:
    _, headers = await account()
    client.delete("/users/me", headers=headers)
    assert client.get("/users/me", headers=headers).status_code == 404


async def test_a_deleted_account_can_no_longer_log_in(client: TestClient) -> None:
    _, headers = await account()
    client.delete("/users/me", headers=headers)
    login = client.post(
        "/auth/login", json={"email": "person@example.com", "password": PASSWORD}
    )
    assert login.status_code == 401


async def test_deleting_one_account_leaves_the_other_alone(
    client: TestClient,
) -> None:
    _, first_headers = await account("first@example.com")
    _, second_headers = await account("second@example.com")
    client.delete("/users/me", headers=first_headers)
    assert client.get("/users/me", headers=second_headers).status_code == 200


def test_a_delete_with_no_token_is_turned_away(client: TestClient) -> None:
    assert client.delete("/users/me").status_code == 401
