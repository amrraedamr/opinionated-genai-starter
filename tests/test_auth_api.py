"""The auth surface: register, log in, refresh, and read yourself."""

from __future__ import annotations

from uuid import uuid4

import pytest


@pytest.fixture(scope="class")
def account(client, api_prefix) -> dict[str, str]:
    """A registered account plus a fresh token pair."""
    credentials = {
        "email": f"{uuid4().hex}@example.com",
        "password": "Str0ng!Passw0rd",
        "firstName": "Test",
        "lastName": "User",
    }
    registered = client.post(f"{api_prefix}/auth/register", json=credentials)
    assert registered.status_code == 200, registered.text

    logged_in = client.post(
        f"{api_prefix}/auth/login",
        json={"email": credentials["email"], "password": credentials["password"]},
    )
    assert logged_in.status_code == 200, logged_in.text
    tokens = logged_in.json()["data"]
    return {**credentials, **tokens}


class TestRegistration:
    def test_a_new_account_is_created(self, client, api_prefix):
        response = client.post(
            f"{api_prefix}/auth/register",
            json={
                "email": f"{uuid4().hex}@example.com",
                "password": "Str0ng!Passw0rd",
                "firstName": "New",
                "lastName": "Person",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["userId"]

    def test_the_same_email_cannot_register_twice(self, client, api_prefix):
        payload = {
            "email": f"{uuid4().hex}@example.com",
            "password": "Str0ng!Passw0rd",
            "firstName": "First",
            "lastName": "Only",
        }
        assert client.post(f"{api_prefix}/auth/register", json=payload).status_code == 200
        assert client.post(f"{api_prefix}/auth/register", json=payload).status_code >= 400

    def test_a_weak_password_is_rejected(self, client, api_prefix):
        response = client.post(
            f"{api_prefix}/auth/register",
            json={"email": f"{uuid4().hex}@example.com", "password": "short"},
        )
        assert response.status_code >= 400


class TestLogin:
    def test_it_returns_a_token_pair(self, account):
        assert account["accessToken"]
        assert account["refreshToken"]

    def test_a_wrong_password_is_unauthorized(self, client, api_prefix, account):
        response = client.post(
            f"{api_prefix}/auth/login",
            json={"email": account["email"], "password": "Wr0ng!Passw0rd"},
        )
        assert response.status_code >= 400

    def test_an_unknown_email_is_unauthorized(self, client, api_prefix):
        response = client.post(
            f"{api_prefix}/auth/login",
            json={"email": f"{uuid4().hex}@example.com", "password": "Str0ng!Passw0rd"},
        )
        assert response.status_code >= 400


class TestRefresh:
    def test_a_refresh_token_buys_a_new_pair(self, client, api_prefix, account):
        response = client.post(
            f"{api_prefix}/auth/refresh",
            json={"refreshToken": account["refreshToken"]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["accessToken"]

    def test_an_access_token_is_not_a_refresh_token(self, client, api_prefix, account):
        response = client.post(
            f"{api_prefix}/auth/refresh",
            json={"refreshToken": account["accessToken"]},
        )
        assert response.status_code >= 400


class TestCurrentUser:
    def test_a_bearer_token_identifies_the_caller(self, client, api_prefix, account):
        response = client.get(
            f"{api_prefix}/users/me",
            headers={"Authorization": f"Bearer {account['accessToken']}"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["email"] == account["email"]

    def test_no_token_is_rejected(self, client, api_prefix):
        assert client.get(f"{api_prefix}/users/me").status_code >= 400

    def test_a_nonsense_token_is_rejected(self, client, api_prefix):
        response = client.get(
            f"{api_prefix}/users/me", headers={"Authorization": "Bearer not-a-token"}
        )
        assert response.status_code >= 400
