"""`routers/auth.py`: the status codes and the wire shape.

The router is thin on purpose, so what is worth pinning is what a client
sees — the envelope, the status a refused login gets, and the fact that an
OAuth callback is a browser redirect rather than a JSON reply.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from starlette.testclient import TestClient

from base.models.definitions.oauth import OAUTH_NONCE_COOKIE
from base.models.definitions.tenancy import DEFAULT_TENANT_CLAIM
from base.models.definitions.tokens import TokenType
from base.services.auth import TokenVerifier
from identity.models.orm.user import UserDTO
from identity.routers.auth import router
from identity.stores.user import UserStore

from .conftest import FakeHttp, client_for, enable_apple

PASSWORD = "Str0ng!pass"


@pytest.fixture
def client(identity_settings: Any) -> Iterator[TestClient]:
    with client_for(router) as test_client:
        yield test_client


def register(client: TestClient, email: str = "person@example.com", **body: Any) -> Any:
    payload: dict[str, Any] = {"email": email, "password": PASSWORD, **body}
    return client.post("/auth/register", json=payload)


def login(client: TestClient, email: str = "person@example.com", **body: Any) -> Any:
    payload: dict[str, Any] = {"email": email, "password": PASSWORD, **body}
    return client.post("/auth/login", json=payload)


# Registering


def test_registering_returns_the_new_user_id(client: TestClient) -> None:
    response = register(client, first_name="Ada", last_name="Lovelace")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["userId"]


async def test_a_registered_password_is_stored_hashed(client: TestClient) -> None:
    register(client)
    user = await UserStore()._aget("person@example.com", column_name="email")
    assert user is not None
    assert user.password_hash is not None
    assert PASSWORD not in user.password_hash


def test_a_user_registered_into_a_tenant_logs_in_as_that_tenant(
    client: TestClient,
) -> None:
    register(client, tenantId="acme")
    token = login(client).json()["data"]["accessToken"]
    claims = TokenVerifier().decode_local(token, expected_type=TokenType.ACCESS)
    assert claims[DEFAULT_TENANT_CLAIM] == "acme"


def test_registering_an_address_twice_is_a_conflict(client: TestClient) -> None:
    register(client)
    assert register(client).status_code == 409


def test_a_password_too_weak_to_accept_is_refused(client: TestClient) -> None:
    assert register(client, password="short").status_code == 422


def test_a_registration_sending_its_names_as_null_succeeds(
    client: TestClient,
) -> None:
    """A JSON client that writes absent fields as `null` is the common one."""
    response = register(client, firstName=None, lastName=None)
    assert response.status_code == 200


def test_an_address_that_is_not_an_address_is_refused(client: TestClient) -> None:
    assert register(client, email="not-an-address").status_code == 422


def test_a_name_with_digits_in_it_is_refused(client: TestClient) -> None:
    assert register(client, first_name="Ada2").status_code == 422


# Logging in


def test_logging_in_returns_a_pair(client: TestClient) -> None:
    register(client)
    body = login(client).json()["data"]
    assert body["accessToken"]
    assert body["refreshToken"]


def test_the_access_token_a_login_returns_names_its_user(
    client: TestClient,
) -> None:
    user_id = register(client).json()["data"]["userId"]
    token = login(client).json()["data"]["accessToken"]
    claims = TokenVerifier().decode_local(token, expected_type=TokenType.ACCESS)
    assert claims["user_id"] == user_id


def test_an_address_nobody_registered_is_unauthorized(client: TestClient) -> None:
    assert login(client, email="stranger@example.com").status_code == 401


def test_a_wrong_password_is_unauthorized(client: TestClient) -> None:
    register(client)
    assert login(client, password="Wr0ng!pass").status_code == 401


async def test_an_account_with_no_password_cannot_be_logged_into(
    client: TestClient,
) -> None:
    """An OAuth-created account has nothing to verify a password against."""
    await UserStore().aregister(UserDTO(email="oauth@example.com", password_hash=None))
    assert login(client, email="oauth@example.com").status_code == 401


def test_a_login_with_no_password_is_refused_before_it_is_checked(
    client: TestClient,
) -> None:
    response = client.post("/auth/login", json={"email": "a@b.com", "password": ""})
    assert response.status_code == 422


async def test_a_deleted_account_cannot_log_in(client: TestClient) -> None:
    register(client)
    user = await UserStore()._aget("person@example.com", column_name="email")
    assert user is not None
    await UserStore().adelete(user.id)
    assert login(client).status_code == 401


# Refreshing


def test_a_refresh_token_buys_a_new_pair(client: TestClient) -> None:
    register(client)
    refresh_token = login(client).json()["data"]["refreshToken"]
    response = client.post("/auth/refresh", json={"refreshToken": refresh_token})
    assert response.status_code == 200
    assert response.json()["data"]["accessToken"]


def test_an_access_token_is_not_a_refresh_token(client: TestClient) -> None:
    register(client)
    access_token = login(client).json()["data"]["accessToken"]
    response = client.post("/auth/refresh", json={"refreshToken": access_token})
    assert response.status_code == 401


def test_a_refresh_token_that_is_nonsense_is_unauthorized(
    client: TestClient,
) -> None:
    response = client.post("/auth/refresh", json={"refreshToken": "nonsense"})
    assert response.status_code == 401


def test_a_refresh_request_with_no_token_is_refused(client: TestClient) -> None:
    assert client.post("/auth/refresh", json={}).status_code == 422


def test_a_refresh_token_longer_than_any_token_is_refused(
    client: TestClient,
) -> None:
    response = client.post("/auth/refresh", json={"refreshToken": "x" * 8193})
    assert response.status_code == 422


# The OAuth front channel


def test_the_provider_list_is_what_this_service_configured(
    client: TestClient,
) -> None:
    response = client.get("/auth/oauth/providers")
    assert response.json()["data"]["providers"] == ["github", "google"]


def test_starting_a_flow_redirects_to_the_provider(
    client: TestClient, google_http: FakeHttp
) -> None:
    response = client.get("/auth/oauth/google", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith("https://accounts.google.com/")


def test_starting_a_flow_for_an_unknown_provider_is_a_bad_request(
    client: TestClient, google_http: FakeHttp
) -> None:
    response = client.get("/auth/oauth/nowhere", follow_redirects=False)
    assert response.status_code == 400


def test_a_callback_redirects_the_browser_on_with_an_exchange_code(
    client: TestClient, google_http: FakeHttp
) -> None:
    start = client.get("/auth/oauth/google", follow_redirects=False)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    response = client.get(
        f"/auth/oauth/google/callback?code=c&state={state}", follow_redirects=False
    )
    assert response.status_code == 302
    location = urlsplit(response.headers["location"])
    assert location.netloc == "app.example.com"
    assert parse_qs(location.query)["code"][0]


def test_a_callback_with_no_code_is_a_bad_request(
    client: TestClient, google_http: FakeHttp
) -> None:
    response = client.get("/auth/oauth/google/callback?state=s", follow_redirects=False)
    assert response.status_code == 400


def test_a_callback_with_no_state_is_a_bad_request(
    client: TestClient, google_http: FakeHttp
) -> None:
    response = client.get("/auth/oauth/google/callback?code=c", follow_redirects=False)
    assert response.status_code == 400


def test_a_callback_from_a_browser_holding_no_cookie_is_a_bad_request(
    client: TestClient, google_http: FakeHttp
) -> None:
    start = client.get("/auth/oauth/google", follow_redirects=False)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    client.cookies.delete(OAUTH_NONCE_COOKIE)
    response = client.get(
        f"/auth/oauth/google/callback?code=c&state={state}", follow_redirects=False
    )
    assert response.status_code == 400


def test_a_callback_may_be_posted_as_a_form(
    client: TestClient, google_http: FakeHttp
) -> None:
    """Apple answers with `response_mode=form_post` rather than a query."""
    start = client.get("/auth/oauth/google", follow_redirects=False)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    response = client.post(
        "/auth/oauth/google/callback",
        data={"code": "c", "state": state},
        follow_redirects=False,
    )
    assert response.status_code == 302


def test_a_posted_callback_with_an_empty_form_is_a_bad_request(
    client: TestClient, google_http: FakeHttp
) -> None:
    response = client.post(
        "/auth/oauth/google/callback", data={}, follow_redirects=False
    )
    assert response.status_code == 400


# Spending the exchange code


def test_an_exchange_code_is_traded_for_a_pair(
    client: TestClient, google_http: FakeHttp
) -> None:
    start = client.get("/auth/oauth/google", follow_redirects=False)
    state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
    callback = client.get(
        f"/auth/oauth/google/callback?code=c&state={state}", follow_redirects=False
    )
    code = parse_qs(urlsplit(callback.headers["location"]).query)["code"][0]
    response = client.post("/auth/oauth/exchange", json={"code": code})
    assert response.status_code == 200
    assert response.json()["data"]["accessToken"]


def test_an_exchange_code_this_service_never_signed_is_a_bad_request(
    client: TestClient,
) -> None:
    response = client.post("/auth/oauth/exchange", json={"code": "invented"})
    assert response.status_code == 400


def test_an_empty_exchange_code_is_refused(client: TestClient) -> None:
    assert client.post("/auth/oauth/exchange", json={"code": ""}).status_code == 422


def test_an_exchange_code_longer_than_any_code_is_refused(
    client: TestClient,
) -> None:
    response = client.post("/auth/oauth/exchange", json={"code": "x" * 2049})
    assert response.status_code == 422


# Apple, whose callback is a form post carrying the user's name once


@pytest.fixture
def apple_client(make_identity_settings: Any, http: FakeHttp) -> Iterator[TestClient]:
    enable_apple(make_identity_settings, http)
    with client_for(router) as test_client:
        yield test_client


def apple_state(client: TestClient) -> str:
    start = client.get("/auth/oauth/apple", follow_redirects=False)
    return parse_qs(urlsplit(start.headers["location"]).query)["state"][0]


async def test_a_posted_user_blob_reaches_the_account_it_describes(
    apple_client: TestClient,
) -> None:
    response = apple_client.post(
        "/auth/oauth/apple/callback",
        data={
            "code": "c",
            "state": apple_state(apple_client),
            "user": '{"name": {"firstName": "Ada", "lastName": "Lovelace"}}',
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    user = await UserStore()._aget("person@example.com", column_name="email")
    assert user is not None
    assert (user.first_name, user.last_name) == ("Ada", "Lovelace")


def test_a_posted_callback_with_no_user_blob_still_completes(
    apple_client: TestClient,
) -> None:
    response = apple_client.post(
        "/auth/oauth/apple/callback",
        data={"code": "c", "state": apple_state(apple_client)},
        follow_redirects=False,
    )
    assert response.status_code == 302
