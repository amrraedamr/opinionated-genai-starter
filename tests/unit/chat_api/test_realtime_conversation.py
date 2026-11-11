"""The conversation route under a real uvicorn server and a real websocket client."""

import asyncio
import json
import socket
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx2 as httpx
import pytest
import uvicorn
import websockets
from websockets.asyncio.client import ClientConnection

from base.app import create_app
from base.models.definitions.realtime import (
    TICKET_SUBPROTOCOL_PREFIX,
    ChannelDelivery,
    RealtimeMessageType,
)
from base.models.view_models.realtime import RealtimeEnvelope
from base.models.view_models.realtime_session import (
    RealtimeInput,
    RealtimeOutput,
    RealtimeTextDelta,
    RealtimeTextInput,
    RealtimeTurnEnd,
)
from base.services.configuration.mixins import (
    RateLimitSettingsMixin,
    RealtimeSettingsMixin,
    ServerSettingsMixin,
    SQLSettingsMixin,
    TenancySettingsMixin,
    TokenIssuingSettingsMixin,
)
from base.services.llm.realtime import AbstractRealtimeSession
from base.services.realtime import ConnectionManager
from base.services.tool_calling import Toolbox
from chat_api import routers
from chat_api.lifespans import realtime_lifespan
from chat_api.routers import realtime as realtime_router
from tests.conftest import MakeSettings
from tests.unit.chat_api.conftest import access_token, bearer

REPLY_TIMEOUT = 5.0


class ScriptedSession(AbstractRealtimeSession):
    """A provider that answers each typed turn with a fixed reply."""

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.sent: list[RealtimeInput] = []
        self._turns: asyncio.Queue[str] = asyncio.Queue()

    @property
    def model(self) -> str:
        return "scripted"

    @property
    def input_sample_rate(self) -> int:
        return 24_000

    def get_service_name(self) -> str:
        return "scripted"

    async def aconnect(self) -> None:
        self.connected = True

    async def asend(self, value: RealtimeInput) -> None:
        self.sent.append(value)
        if isinstance(value, RealtimeTextInput):
            await self._turns.put(f"you said {value.text}")

    async def areceive(self) -> AsyncIterator[RealtimeOutput]:
        if not self.connected:
            raise RuntimeError("Realtime session used before aconnect")
        while True:
            reply = await self._turns.get()
            yield RealtimeTextDelta(text=reply)
            yield RealtimeTurnEnd()

    async def asend_tool_result(self, call_id: str, result: str) -> None:
        return None

    async def ainterrupt(self) -> None:
        return None

    async def aclose(self) -> None:
        self.closed = True


class Server:
    """A running app, reachable over HTTP and websocket."""

    def __init__(self, port: int) -> None:
        self.http = f"http://127.0.0.1:{port}"
        self.ws = f"ws://127.0.0.1:{port}"

    async def aticket(self, user: uuid.UUID, device: str = "laptop") -> str:
        async with httpx.AsyncClient(base_url=self.http) as client:
            response = await client.post(
                "/api/realtime/tickets",
                json={"deviceId": device},
                headers={"Authorization": f"Bearer {access_token(user)}"},
            )
        assert response.status_code == 200, response.text
        return str(response.json()["ticket"])

    def connect(self, ticket: str) -> Any:
        return websockets.connect(
            f"{self.ws}/api/realtime/conversation",
            subprotocols=[f"{TICKET_SUBPROTOCOL_PREFIX}{ticket}"],  # type: ignore[list-item]
        )


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[ScriptedSession]:
    """Every provider session the route opens, in order."""
    opened: list[ScriptedSession] = []

    def build(toolbox: Toolbox | None = None) -> ScriptedSession:
        del toolbox
        opened.append(ScriptedSession())
        return opened[-1]

    monkeypatch.setattr(realtime_router, "build_realtime_session", build)
    return opened


@pytest.fixture
def tenancy() -> bool:
    """Off, as most of these run; a test that needs it overrides this."""
    return False


@pytest.fixture
async def server(
    make_settings: MakeSettings,
    sqlite_path: Any,
    key_paths: tuple[str, str],
    tenancy: bool,
) -> AsyncIterator[Server]:
    private_path, public_path = key_paths
    settings = make_settings(
        ServerSettingsMixin,
        SQLSettingsMixin,
        TokenIssuingSettingsMixin,
        RealtimeSettingsMixin,
        RateLimitSettingsMixin,
        TenancySettingsMixin,
        tenancy_enabled=tenancy,
        service_name="chat_api",
        sql_backend="sqlite",
        sqlite_db_path=str(sqlite_path),
        db_ssl_enabled=False,
        auth_private_key_path=private_path,
        auth_public_key_path=public_path,
    )
    app = create_app(
        settings=settings,
        routers=routers,
        prefix="/api",
        lifespan_hooks=[realtime_lifespan],
    )
    port = free_port()
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    serving = asyncio.create_task(uvicorn_server.serve())
    while not uvicorn_server.started:
        await asyncio.sleep(0.02)
    try:
        yield Server(port)
    finally:
        uvicorn_server.should_exit = True
        await asyncio.wait_for(serving, timeout=10)


async def frames_until_turn_end(ws: ClientConnection) -> list[RealtimeEnvelope]:
    received: list[RealtimeEnvelope] = []
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=REPLY_TIMEOUT)
        frame = RealtimeEnvelope.model_validate_json(raw)
        received.append(frame)
        if frame.payload.get("turnComplete"):
            return received


def say(text: str) -> str:
    return json.dumps({"type": "message", "payload": {"text": text}})


async def eventually(condition: Any, timeout: float = 2.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.02)
    return bool(condition())


async def test_a_typed_question_is_answered_through_the_route(
    server: Server, sessions: list[ScriptedSession]
) -> None:
    ticket = await server.aticket(uuid.uuid4())
    async with server.connect(ticket) as ws:
        await ws.send(say("hello"))
        frames = await frames_until_turn_end(ws)

    assert [f.payload["text"] for f in frames if "text" in f.payload] == [
        "you said hello"
    ]
    assert sessions[0].connected


async def test_the_provider_session_is_closed_when_the_client_leaves(
    server: Server, sessions: list[ScriptedSession]
) -> None:
    ticket = await server.aticket(uuid.uuid4())
    async with server.connect(ticket) as ws:
        await ws.send(say("hello"))
        await frames_until_turn_end(ws)

    assert await eventually(lambda: sessions[0].closed)


async def test_a_ticket_opens_exactly_one_socket(
    server: Server, sessions: list[ScriptedSession]
) -> None:
    ticket = await server.aticket(uuid.uuid4())
    async with server.connect(ticket):
        pass

    with pytest.raises(websockets.exceptions.InvalidStatus):
        async with server.connect(ticket):
            pass


async def test_a_durable_message_reaches_a_device_that_resumes(
    server: Server, sessions: list[ScriptedSession]
) -> None:
    user = uuid.uuid4()
    await ConnectionManager().apublish_to_user(
        str(user),
        RealtimeEnvelope(
            type=RealtimeMessageType.MESSAGE, payload={"note": "while you were away"}
        ),
        ChannelDelivery.DURABLE,
    )

    ticket = await server.aticket(user)
    async with server.connect(ticket) as ws:
        await ws.send(json.dumps({"type": "resume", "seq": 0}))
        missed = RealtimeEnvelope.model_validate_json(
            await asyncio.wait_for(ws.recv(), timeout=REPLY_TIMEOUT)
        )

    assert missed.payload == {"note": "while you were away"}
    assert missed.seq == 1


async def test_a_message_published_on_another_replica_reaches_the_socket(
    server: Server, sessions: list[ScriptedSession]
) -> None:
    """Fan-out runs only if the lifespan started the manager's listener."""
    user = uuid.uuid4()
    ticket = await server.aticket(user)
    async with server.connect(ticket) as ws:
        assert await eventually(lambda: bool(ConnectionManager().devices_of(str(user))))
        await ConnectionManager().apublish_to_user(
            str(user),
            RealtimeEnvelope(type=RealtimeMessageType.MESSAGE, payload={"ping": 1}),
        )
        pushed = RealtimeEnvelope.model_validate_json(
            await asyncio.wait_for(ws.recv(), timeout=REPLY_TIMEOUT)
        )

    assert pushed.payload == {"ping": 1}


async def test_a_notification_you_send_yourself_reaches_your_open_socket(
    server: Server, sessions: list[ScriptedSession]
) -> None:
    user = uuid.uuid4()
    ticket = await server.aticket(user)
    async with server.connect(ticket) as ws:
        async with httpx.AsyncClient(base_url=server.http) as client:
            response = await client.post(
                "/api/realtime/notifications",
                json={"payload": {"note": "hi"}},
                headers=bearer(user),
            )
        assert response.status_code == 200, response.text
        seq = response.json()["data"]["seq"]
        frame = RealtimeEnvelope.model_validate_json(
            await asyncio.wait_for(ws.recv(), timeout=REPLY_TIMEOUT)
        )
    assert frame.payload == {"note": "hi"}
    assert frame.channel == "notifications"
    assert frame.seq == seq == 1


async def test_a_notification_waits_for_a_device_that_was_offline(
    server: Server, sessions: list[ScriptedSession]
) -> None:
    user = uuid.uuid4()
    async with httpx.AsyncClient(base_url=server.http) as client:
        await client.post(
            "/api/realtime/notifications",
            json={"payload": {"note": "while you were away"}},
            headers=bearer(user),
        )
    async with server.connect(await server.aticket(user)) as ws:
        await ws.send(json.dumps({"type": "resume", "seq": 0}))
        missed = RealtimeEnvelope.model_validate_json(
            await asyncio.wait_for(ws.recv(), timeout=REPLY_TIMEOUT)
        )
    assert missed.payload == {"note": "while you were away"}


async def test_presence_shows_an_open_socket(
    server: Server, sessions: list[ScriptedSession]
) -> None:
    user = uuid.uuid4()

    async def online() -> bool:
        async with httpx.AsyncClient(base_url=server.http) as client:
            response = await client.get("/api/realtime/presence", headers=bearer(user))
        assert response.status_code == 200, response.text
        return bool(response.json()["data"]["online"])

    assert not await online()
    async with server.connect(await server.aticket(user)):
        assert await online()


@pytest.mark.parametrize("tenancy", [True])
async def test_a_notification_reaches_the_inbox_the_socket_reads_with_tenancy_on(
    server: Server, sessions: list[ScriptedSession]
) -> None:
    """The route and the socket must agree on the tenant, or resume reads an empty inbox."""
    user = uuid.uuid4()
    headers = {"Authorization": f"Bearer {access_token(user, tid='acme')}"}
    async with httpx.AsyncClient(base_url=server.http) as client:
        await client.post(
            "/api/realtime/notifications",
            json={"payload": {"note": "while you were away"}},
            headers=headers,
        )
        ticket = (
            await client.post("/api/realtime/tickets", json={}, headers=headers)
        ).json()["ticket"]
    async with server.connect(ticket) as ws:
        await ws.send(json.dumps({"type": "resume", "seq": 0}))
        missed = RealtimeEnvelope.model_validate_json(
            await asyncio.wait_for(ws.recv(), timeout=REPLY_TIMEOUT)
        )
    assert missed.payload == {"note": "while you were away"}
