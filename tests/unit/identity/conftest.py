"""What the identity service needs before any of it can be exercised.

Its settings are one module-level object that several modules imported by
value, so a test's settings only take effect once that name is rebound
wherever it landed. Storage is a throwaway SQLite file, as in `tests/unit/sql`:
the unique email, the outbox row and the soft delete are engine-independent,
and a container would only make them slower.

Nothing here reaches a network. The provider registry is the real one; only
the HTTP client under it is replaced, so a test exercises the parsing and the
error branches rather than agreeing with itself.
"""

from __future__ import annotations

import importlib
import json
import pathlib
from collections.abc import Iterator
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import APIRouter, FastAPI
from jwt import encode
from pydantic_settings import SettingsConfigDict
from starlette.testclient import TestClient

from base.models.view_models.http import HttpResponse
from base.services.common.meta import KeyedSingleton, Singleton, purge_all
from base.services.configuration import configure
from base.services.http import HttpRequestService
from base.services.sql.service import SQLService
from base.tools.exception_handlers import AppExceptionHandler
from identity.settings import IdentitySettings

HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

SETTINGS_HOLDERS: tuple[str, ...] = (
    "identity.settings",
    "identity.routers.auth",
    "identity.services.auth",
    "identity.services.auth.oauth",
)
"""Every module that did `from identity.settings import settings`."""

GOOGLE_REDIRECT: str = "https://identity.example.com/auth/oauth/google/callback"
GITHUB_REDIRECT: str = "https://identity.example.com/auth/oauth/github/callback"
APPLE_REDIRECT: str = "https://identity.example.com/auth/oauth/apple/callback"
GOOGLE_DISCOVERY: dict[str, str] = {
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
}
APPLE_DISCOVERY: dict[str, str] = {
    "authorization_endpoint": "https://appleid.apple.com/auth/authorize",
    "token_endpoint": "https://appleid.apple.com/auth/token",
}


class IsolatedIdentitySettings(IdentitySettings):
    """Identity's own settings, with a developer's `.env` kept out."""

    model_config = SettingsConfigDict(
        env_file=None, extra="ignore", populate_by_name=True
    )


def _abandon_engines() -> None:
    """Drop pooled connections without needing a loop that has already closed."""
    for instance in list(Singleton._instances.values()):
        if isinstance(instance, SQLService):
            instance.dispose()
    for client in list(KeyedSingleton._keyed_instances.values()):
        engine = getattr(client, "_async_engine", None)
        if engine is not None:
            engine.sync_engine.dispose(close=False)


@pytest.fixture(autouse=True)
def _fresh_services() -> Iterator[None]:
    """A service built in one test never answers the next."""
    purge_all()
    yield
    _abandon_engines()
    purge_all()


@pytest.fixture(scope="session")
def rsa_pem() -> tuple[str, str]:
    """One RSA pair for the whole session; generating it is the slow part."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_pem, public_pem


@pytest.fixture
def keys(rsa_pem: tuple[str, str], tmp_path: pathlib.Path) -> tuple[str, str]:
    """The session's RSA pair, written where settings can point at it."""
    private_pem, public_pem = rsa_pem
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_text(private_pem)
    public_path.write_text(public_pem)
    return str(private_path), str(public_path)


@pytest.fixture
def sqlite_path(unique: str) -> Iterator[pathlib.Path]:
    """A SQLite file of this test's own, removed afterwards."""
    path = HERE / f"identity_{unique}.db"
    yield path
    for suffix in ("", "-wal", "-shm"):
        candidate = pathlib.Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


def _create_schema() -> None:
    """The identity tables on whichever database the settings now point at."""
    from identity.models.orm import (  # noqa: F401
        IdentityBase,
        outbox,
        user,
        user_identity,
    )

    with SQLService()._get_session() as session:
        IdentityBase.metadata.create_all(session.get_bind())


@pytest.fixture
def make_identity_settings(
    monkeypatch: pytest.MonkeyPatch, keys: tuple[str, str], sqlite_path: pathlib.Path
) -> Any:
    """Register settings for one test and give it an empty identity schema."""
    private_path, public_path = keys

    def _make(**overrides: Any) -> IdentitySettings:
        values: dict[str, Any] = {
            "auth_private_key_path": private_path,
            "auth_public_key_path": public_path,
            "sql_backend": "sqlite",
            "sqlite_db_path": str(sqlite_path),
            "db_ssl_enabled": False,
            "redis_url": "",
            "oauth_providers": "google,github",
            "oauth_google_client_id": "google-client",
            "oauth_google_client_secret": "google-secret",
            "oauth_google_redirect_uri": GOOGLE_REDIRECT,
            "oauth_github_client_id": "github-client",
            "oauth_github_client_secret": "github-secret",
            "oauth_github_redirect_uri": GITHUB_REDIRECT,
            "oauth_success_redirect": "https://app.example.com/auth/callback",
            **overrides,
        }
        settings = IsolatedIdentitySettings(**values)
        for name in SETTINGS_HOLDERS:
            monkeypatch.setattr(importlib.import_module(name), "settings", settings)
        configure(settings)
        _create_schema()
        return settings

    return _make


@pytest.fixture
def identity_settings(make_identity_settings: Any) -> IdentitySettings:
    """Google and GitHub enabled, keys on disk, an empty SQLite schema."""
    return make_identity_settings()


class FakeHttp:
    """Answers the OAuth registry from a table keyed by a URL fragment.

    The registry builds the request — the token body, the bearer header — so
    what it sent is kept rather than discarded.
    """

    def __init__(self) -> None:
        self.routes: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []

    def route(self, fragment: str, reply: Any) -> None:
        self.routes[fragment] = reply

    def json_route(self, fragment: str, body: dict[str, Any]) -> None:
        self.route(fragment, HttpResponse.ok(200, body))

    def fail_route(self, fragment: str, status_code: int = 502) -> None:
        self.route(fragment, HttpResponse.fail(RuntimeError("no"), status_code))

    async def acall_endpoint(self, url: str, method: str = "GET", **kwargs: Any) -> Any:
        self.calls.append({"url": url, "method": method, **kwargs})
        # A fragment reaching the end of the URL wins, so `/user/emails` is
        # not answered by the route standing in for `/user`.
        matches = [f for f in self.routes if url.endswith(f)] or [
            f for f in self.routes if f in url
        ]
        if not matches:
            return HttpResponse.fail(AssertionError(f"unrouted: {url}"), 404)
        return self.routes[max(matches, key=len)]

    def sent_to(self, fragment: str) -> dict[str, Any]:
        for call in reversed(self.calls):
            if fragment in call["url"]:
                return call
        raise AssertionError(f"nothing was sent to {fragment}")

    def called(self, fragment: str) -> int:
        return sum(1 for call in self.calls if fragment in call["url"])


@pytest.fixture(autouse=True)
def http() -> FakeHttp:
    """The registry's HTTP client, replaced before the real one is built.

    Autouse because a unit test must not build the real one at all: the
    registry constructs it the moment an `OAuthService` exists.
    """
    fake = FakeHttp()
    HttpRequestService.replace(fake)
    return fake


@pytest.fixture
def google_http(http: FakeHttp) -> FakeHttp:
    """Google discovered, exchanging a code for an id_token with a verified email."""
    http.json_route("openid-configuration", GOOGLE_DISCOVERY)
    http.json_route(
        "oauth2.googleapis.com/token",
        {"access_token": "at", "id_token": id_token(), "token_type": "Bearer"},
    )
    return http


def id_token(**claims: Any) -> str:
    """An id_token the registry will read; it never verifies the signature."""
    payload: dict[str, Any] = {
        "sub": "google-subject",
        "email": "person@example.com",
        "email_verified": True,
        "given_name": "Ada",
        "family_name": "Lovelace",
        **claims,
    }
    return encode(payload, key="x" * 32, algorithm="HS256")


def enable_apple(make_identity_settings: Any, http: FakeHttp) -> None:
    """Apple alone, with a pre-generated secret and its endpoints routed."""
    make_identity_settings(
        oauth_providers="apple",
        oauth_apple_client_id="com.example.service",
        oauth_apple_client_secret="apple-secret",
        oauth_apple_redirect_uri=APPLE_REDIRECT,
    )
    http.json_route(
        "appleid.apple.com/.well-known/openid-configuration", APPLE_DISCOVERY
    )
    http.json_route(
        "appleid.apple.com/auth/token",
        {"access_token": "at", "id_token": id_token(sub="apple-subject")},
    )


def github_emails(email: str, verified: bool = True, primary: bool = True) -> str:
    """GitHub's `/user/emails` body, which the registry reads raw."""
    return json.dumps([{"email": email, "primary": primary, "verified": verified}])


def build_app(*routers: APIRouter) -> FastAPI:
    """An app wired the way `create_app` wires one, minus the capabilities.

    The exception handlers are the point: without them an `ErrorsException`
    is an unhandled 500 rather than the status the error names.
    """
    app = FastAPI()
    AppExceptionHandler().register(app)
    for router in routers:
        app.include_router(router)
    return app


def client_for(*routers: APIRouter) -> TestClient:
    """A client over those routers, on an origin a `__Host-` cookie accepts."""
    return TestClient(build_app(*routers), base_url="https://identity.example.com")
