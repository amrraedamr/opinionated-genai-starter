"""Token verification and ownership, without any identity service.

chat_api does not own users. It only needs a token signed by a key it trusts,
so these tests mint tokens directly and never call identity.
"""

import uuid
from datetime import UTC, datetime

import pytest
from jwt import encode


def sign(key: str, claims: dict, *, ttl_seconds: int = 900) -> str:
    now = datetime.now(UTC)
    return encode(
        {**claims, "iat": now, "exp": now.timestamp() + ttl_seconds},
        key=key,
        algorithm="RS256",
    )


@pytest.fixture
def issue_token(client, signing_key):
    def _issue(user_id: uuid.UUID | None = None) -> str:
        user_id = user_id or uuid.uuid4()
        return sign(
            signing_key,
            {"user_id": str(user_id), "email": f"{user_id}@example.com", "type": "access"},
        )

    return _issue


class TestHealth:
    def test_livez_does_not_touch_dependencies(self, client):
        body = client.get("/livez").json()
        assert body["status"] == "ok"
        assert body["checks"] == []

    def test_readyz_reports_capability_probes(self, client):
        body = client.get("/readyz").json()
        assert body["status"] == "ok"
        assert {check["name"] for check in body["checks"]} >= {"prompts"}


class TestTokenVerification:
    def test_a_token_signed_by_the_trusted_key_is_accepted(
        self, client, api_prefix, issue_token
    ):
        response = client.post(
            f"{api_prefix}/conversations",
            headers={"Authorization": f"Bearer {issue_token()}"},
            json={"title": "hello"},
        )
        assert response.status_code == 200, response.text

    def test_unauthenticated_request_is_rejected(self, client, api_prefix):
        assert client.post(f"{api_prefix}/conversations", json={}).status_code in (401, 403)

    def test_tampered_signature_is_rejected(self, client, api_prefix, issue_token):
        tampered = issue_token()[:-3] + "xyz"
        response = client.post(
            f"{api_prefix}/conversations",
            headers={"Authorization": f"Bearer {tampered}"},
            json={"title": "hello"},
        )
        assert response.status_code == 401

    def test_a_refresh_token_is_not_accepted_as_an_access_token(
        self, client, api_prefix, signing_key
    ):
        refresh = sign(signing_key, {"user_id": str(uuid.uuid4()), "type": "refresh"})
        response = client.post(
            f"{api_prefix}/conversations",
            headers={"Authorization": f"Bearer {refresh}"},
            json={"title": "hello"},
        )
        assert response.status_code == 401


class TestOwnership:
    def test_a_user_cannot_read_another_users_conversation(
        self, client, api_prefix, issue_token
    ):
        owner_token = issue_token()
        intruder_token = issue_token()

        created = client.post(
            f"{api_prefix}/conversations",
            headers={"Authorization": f"Bearer {owner_token}"},
            json={"title": "private"},
        )
        assert created.status_code == 200, created.text
        conversation_id = created.json()["data"]["conversationId"]

        mine = client.get(
            f"{api_prefix}/conversations/{conversation_id}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        assert mine.status_code == 200

        theirs = client.get(
            f"{api_prefix}/conversations/{conversation_id}",
            headers={"Authorization": f"Bearer {intruder_token}"},
        )
        assert theirs.status_code >= 400
