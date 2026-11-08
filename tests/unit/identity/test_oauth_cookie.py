"""The cookie that binds an OAuth callback to one browser.

Its attributes are the security property, so they are read off a real
response rather than from the source.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from base.models.definitions.oauth import OAUTH_NONCE_COOKIE


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, make_settings: Any) -> TestClient:
    """The real route, with only the OAuth service replaced."""
    import identity.routers.auth as auth_module

    class FakeOAuthService:
        async def abuild_authorize_redirect(self, provider: str) -> tuple[str, str]:
            return "https://provider/authorize?state=abc", "the-nonce"

        async def ahandle_callback(self, provider: str, **kwargs: Any) -> str:
            assert kwargs["nonce"] == "the-nonce", (
                "the cookie never reached the service"
            )
            return "exchange-code"

        def success_redirect_url(self, exchange_code: str) -> str:
            return f"https://app/done?code={exchange_code}"

    monkeypatch.setattr(auth_module, "OAuthService", FakeOAuthService)
    app = FastAPI()
    app.include_router(auth_module.router)
    return TestClient(app, base_url="https://identity.example.com")


def test_the_cookie_is_host_only() -> None:
    """`__Host-` forbids a `Domain`, so a sibling subdomain cannot plant one."""
    assert OAUTH_NONCE_COOKIE.startswith("__Host-")


def test_starting_a_flow_hands_the_browser_its_binding(client: TestClient) -> None:
    response = client.get("/auth/oauth/google", follow_redirects=False)

    assert response.status_code == 302
    header = response.headers["set-cookie"]
    assert OAUTH_NONCE_COOKIE in header
    assert "HttpOnly" in header
    assert "Secure" in header
    assert "SameSite=lax" in header.replace("samesite", "SameSite")
    assert "Path=/" in header
    # `Strict` would strip the cookie from the provider's redirect back.
    assert "SameSite=strict" not in header.lower()


def test_the_callback_sends_the_binding_on_to_the_service(
    client: TestClient,
) -> None:
    """The assertion lives in the fake: a missing cookie fails the request."""
    client.cookies.set(OAUTH_NONCE_COOKIE, "the-nonce")
    response = client.get(
        "/auth/oauth/google/callback?code=c&state=abc", follow_redirects=False
    )
    assert response.status_code == 302


def test_the_binding_is_cleared_once_it_has_been_spent(client: TestClient) -> None:
    client.cookies.set(OAUTH_NONCE_COOKIE, "the-nonce")
    response = client.get(
        "/auth/oauth/google/callback?code=c&state=abc", follow_redirects=False
    )
    assert OAUTH_NONCE_COOKIE in response.headers.get("set-cookie", "")
