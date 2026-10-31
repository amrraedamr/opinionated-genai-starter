"""Conversations belong to the caller, and only to the caller."""

from __future__ import annotations

from uuid import uuid4

import pytest


def _register_and_login(client, api_prefix) -> str:
    email = f"{uuid4().hex}@example.com"
    password = "Str0ng!Passw0rd"
    client.post(
        f"{api_prefix}/auth/register",
        json={"email": email, "password": password, "firstName": "A", "lastName": "B"},
    )
    tokens = client.post(
        f"{api_prefix}/auth/login", json={"email": email, "password": password}
    ).json()["data"]
    return tokens["accessToken"]


@pytest.fixture(scope="class")
def owner(client, api_prefix) -> dict[str, str]:
    return {"Authorization": f"Bearer {_register_and_login(client, api_prefix)}"}


@pytest.fixture(scope="class")
def stranger(client, api_prefix) -> dict[str, str]:
    return {"Authorization": f"Bearer {_register_and_login(client, api_prefix)}"}


@pytest.fixture
def conversation_id(client, api_prefix, owner) -> str:
    response = client.post(
        f"{api_prefix}/conversations",
        json={"title": "Planning", "metadata": {"topic": "penguins"}},
        headers=owner,
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]["conversationId"]


class TestCreate:
    def test_it_returns_an_id(self, conversation_id):
        assert conversation_id

    def test_it_defaults_the_title(self, client, api_prefix, owner):
        created = client.post(f"{api_prefix}/conversations", json={}, headers=owner)
        assert created.status_code == 200, created.text
        fetched = client.get(
            f"{api_prefix}/conversations/{created.json()['data']['conversationId']}",
            headers=owner,
        )
        assert fetched.json()["data"]["title"] == "New Chat"

    def test_it_requires_a_caller(self, client, api_prefix):
        assert client.post(f"{api_prefix}/conversations", json={}).status_code >= 400


class TestRead:
    def test_the_owner_sees_what_they_stored(
        self, client, api_prefix, owner, conversation_id
    ):
        response = client.get(
            f"{api_prefix}/conversations/{conversation_id}", headers=owner
        )
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["title"] == "Planning"
        assert data["metadata"] == {"topic": "penguins"}

    def test_a_stranger_gets_nothing(
        self, client, api_prefix, stranger, conversation_id
    ):
        response = client.get(
            f"{api_prefix}/conversations/{conversation_id}", headers=stranger
        )
        assert response.status_code >= 400

    def test_an_unknown_id_is_not_found(self, client, api_prefix, owner):
        response = client.get(f"{api_prefix}/conversations/{uuid4()}", headers=owner)
        assert response.status_code >= 400


class TestEdit:
    def test_the_owner_can_rename_it(
        self, client, api_prefix, owner, conversation_id
    ):
        edited = client.patch(
            f"{api_prefix}/conversations/{conversation_id}",
            json={"title": "Renamed"},
            headers=owner,
        )
        assert edited.status_code == 200, edited.text

        fetched = client.get(
            f"{api_prefix}/conversations/{conversation_id}", headers=owner
        )
        assert fetched.json()["data"]["title"] == "Renamed"

    def test_a_stranger_cannot_rename_it(
        self, client, api_prefix, stranger, conversation_id
    ):
        response = client.patch(
            f"{api_prefix}/conversations/{conversation_id}",
            json={"title": "Hijacked"},
            headers=stranger,
        )
        assert response.status_code >= 400


class TestDelete:
    def test_the_owner_can_delete_it(
        self, client, api_prefix, owner, conversation_id
    ):
        assert (
            client.delete(
                f"{api_prefix}/conversations/{conversation_id}", headers=owner
            ).status_code
            == 200
        )
        assert (
            client.get(
                f"{api_prefix}/conversations/{conversation_id}", headers=owner
            ).status_code
            >= 400
        )

    def test_deleting_twice_is_not_found(
        self, client, api_prefix, owner, conversation_id
    ):
        client.delete(f"{api_prefix}/conversations/{conversation_id}", headers=owner)
        again = client.delete(
            f"{api_prefix}/conversations/{conversation_id}", headers=owner
        )
        assert again.status_code >= 400

    def test_a_stranger_cannot_delete_it(
        self, client, api_prefix, stranger, conversation_id
    ):
        response = client.delete(
            f"{api_prefix}/conversations/{conversation_id}", headers=stranger
        )
        assert response.status_code >= 400
