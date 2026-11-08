"""Fixtures for the chat_api service's unit tests.

The service is exercised the way it is deployed — real routers behind the
real `create_app`, real token verification, real stores — with only the two
things a unit test may not have replaced: the database is a throwaway SQLite
file, and the agent is a stand-in that never reaches a provider.

Every service here is a singleton of some kind, and an async one caches an
engine bound to the event loop it was built on, so the registries are emptied
around each test and pooled connections are abandoned rather than closed.
"""

from __future__ import annotations

import pathlib
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from base.app import create_app
from base.models.view_models.streams import StreamEnd, StreamMessage
from base.services.auth import TokenIssuer
from base.services.common.meta import KeyedSingleton, Singleton, purge_all
from base.services.configuration.mixins import (
    ServerSettingsMixin,
    SQLSettingsMixin,
    TokenIssuingSettingsMixin,
)
from base.services.execution_context import ExecutionContext
from base.services.sql.service import SQLService
from chat_api import routers

# The DTO modules are imported for their side effect: a table absent from the
# metadata is a table `create_all` never builds, and the foreign key into it
# never resolves.
from chat_api.models.orm import ChatBase
from chat_api.models.orm.chat_history_message import ChatHistoryMessageDTO  # noqa: F401
from chat_api.models.orm.conversation import ConversationDTO  # noqa: F401
from chat_api.models.orm.outbox import OutboxEventDTO  # noqa: F401
from chat_api.models.orm.prompt import ActivePromptDTO, PromptDTO  # noqa: F401
from chat_api.routers import conversation as conversation_router

HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent


def _abandon_engines() -> None:
    """Drop pooled connections without touching a loop that has already closed."""
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


@pytest.fixture
def sqlite_path(unique: str) -> Iterator[pathlib.Path]:
    """A SQLite file of this test's own, removed afterwards."""
    path = HERE / f"chat_{unique}.db"
    yield path
    for suffix in ("", "-wal", "-shm"):
        candidate = pathlib.Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


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
def key_paths(rsa_pem: tuple[str, str], tmp_path: pathlib.Path) -> tuple[str, str]:
    """The session's RSA pair, written where settings can point at it."""
    private_pem, public_pem = rsa_pem
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_text(private_pem)
    public_path.write_text(public_pem)
    return str(private_path), str(public_path)


@pytest.fixture
def chat_settings(
    make_settings: Any, sqlite_path: pathlib.Path, key_paths: tuple[str, str]
) -> Any:
    """Registered settings covering every subsystem this service reads."""
    private_path, public_path = key_paths
    return make_settings(
        ServerSettingsMixin,
        SQLSettingsMixin,
        TokenIssuingSettingsMixin,
        service_name="chat_api",
        sql_backend="sqlite",
        sqlite_db_path=str(sqlite_path),
        db_ssl_enabled=False,
        auth_private_key_path=private_path,
        auth_public_key_path=public_path,
    )


@pytest.fixture
def tables(chat_settings: Any) -> None:
    """Every table chat_api owns, created in this test's SQLite file."""
    with SQLService()._get_session() as session:
        ChatBase.metadata.create_all(session.get_bind())


@pytest.fixture
def user_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def other_user_id() -> uuid.UUID:
    return uuid.uuid4()


def access_token(user_id: uuid.UUID) -> str:
    """A genuine signed access token asserting this user."""
    return TokenIssuer().issue_access({"user_id": str(user_id)})


def bearer(user_id: uuid.UUID) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token(user_id)}"}


@pytest.fixture
def auth(tables: None, user_id: uuid.UUID) -> dict[str, str]:
    """Headers for the user the tests treat as the owner."""
    return bearer(user_id)


@pytest.fixture
def other_auth(tables: None, other_user_id: uuid.UUID) -> dict[str, str]:
    """Headers for somebody else entirely."""
    return bearer(other_user_id)


class StubAgent:
    """Answers like `TestAgent` without reaching a provider.

    A streaming run is the provider's job in production, so the stub does what
    an adapter does: it sends the answer and closes the stream.
    """

    answer: str = "the answer"
    failure: Exception | None = None

    async def arun(self) -> None:
        context = ExecutionContext()
        if self.failure is not None:
            raise self.failure
        context.flow_variables.final_answer = self.answer
        if context.streaming and context.stream_iterator is not None:
            await context.stream_iterator.asend(StreamMessage(data=self.answer))
            await context.stream_iterator.asend(StreamEnd())


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> type[StubAgent]:
    """The router's agent, replaced by one the test can steer."""
    stub = type("StubAgent", (StubAgent,), {})
    monkeypatch.setattr(conversation_router, "TestAgent", stub)
    return stub


@pytest.fixture
def app(chat_settings: Any, tables: None) -> FastAPI:
    """The service's own application: real routers, real middleware, real guards."""
    return create_app(settings=chat_settings, routers=routers, prefix="/api")


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Entered, so the lifespan runs and one event loop outlives each request.

    A streaming turn finishes its work in a background task after the response
    has been read; a client that tore its loop down per request would cancel it.
    """
    with TestClient(app) as test_client:
        yield test_client


def conversation_of(client: TestClient, auth: dict[str, str], **body: Any) -> str:
    """Create a conversation over the API and return its id."""
    response = client.post("/api/conversations", json=body or {}, headers=auth)
    assert response.status_code == 200, response.text
    return str(response.json()["data"]["conversationId"])
