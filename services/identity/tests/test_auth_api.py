"""The authentication surface identity owns."""

import uuid


def register_and_login(client, api_prefix: str) -> dict[str, str]:
    email = f"{uuid.uuid4().hex}@example.com"
    password = "Str0ng!Passw0rd"
    registered = client.post(
        f"{api_prefix}/auth/register",
        json={"email": email, "password": password, "firstName": "A", "lastName": "B"},
    )
    assert registered.status_code == 200, registered.text
    logged_in = client.post(
        f"{api_prefix}/auth/login", json={"email": email, "password": password}
    )
    assert logged_in.status_code == 200, logged_in.text
    return logged_in.json()["data"]


class TestHealth:
    def test_livez(self, client):
        assert client.get("/livez").json()["status"] == "ok"

    def test_readyz(self, client):
        assert client.get("/readyz").status_code == 200


class TestAuthentication:
    def test_round_trip_issues_a_usable_token(self, client, api_prefix):
        tokens = register_and_login(client, api_prefix)
        response = client.get(
            f"{api_prefix}/users/me",
            headers={"Authorization": f"Bearer {tokens['accessToken']}"},
        )
        assert response.status_code == 200

    def test_refresh_returns_a_new_pair(self, client, api_prefix):
        tokens = register_and_login(client, api_prefix)
        response = client.post(
            f"{api_prefix}/auth/refresh", json={"refreshToken": tokens["refreshToken"]}
        )
        assert response.status_code == 200

    def test_unauthenticated_request_is_rejected(self, client, api_prefix):
        assert client.get(f"{api_prefix}/users/me").status_code in (401, 403)

    def test_access_token_is_not_a_refresh_token(self, client, api_prefix):
        tokens = register_and_login(client, api_prefix)
        response = client.post(
            f"{api_prefix}/auth/refresh", json={"refreshToken": tokens["accessToken"]}
        )
        assert response.status_code == 401

    def test_tampered_signature_is_rejected(self, client, api_prefix):
        tokens = register_and_login(client, api_prefix)
        tampered = tokens["accessToken"][:-3] + "xyz"
        response = client.get(
            f"{api_prefix}/users/me", headers={"Authorization": f"Bearer {tampered}"}
        )
        assert response.status_code == 401

    def test_duplicate_registration_is_rejected(self, client, api_prefix):
        payload = {
            "email": f"{uuid.uuid4().hex}@example.com",
            "password": "Str0ng!Passw0rd",
            "firstName": "A",
            "lastName": "B",
        }
        assert client.post(f"{api_prefix}/auth/register", json=payload).status_code == 200
        assert client.post(f"{api_prefix}/auth/register", json=payload).status_code >= 400


class TestOAuth:
    def test_providers_endpoint_responds(self, client, api_prefix):
        assert client.get(f"{api_prefix}/auth/oauth/providers").status_code == 200

    def test_a_garbage_exchange_code_is_rejected(self, client, api_prefix):
        response = client.post(f"{api_prefix}/auth/oauth/exchange", json={"code": "garbage"})
        assert response.status_code >= 400
