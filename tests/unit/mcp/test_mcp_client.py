"""The MCP client against a real server in a process of its own."""

import asyncio
import json
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx2
import pytest
from mcp import types

from base.capabilities.mcp_clients import McpClients
from base.clients.credentials import StaticTokenProvider
from base.models.definitions.guardrails import GuardrailStage
from base.models.definitions.mcp import McpTransport
from base.models.view_models.chat_history import StoredToolCall, StoredToolCallFunction
from base.models.view_models.exceptions import (
    McpConfigError,
    McpConnectionError,
    McpTimeoutError,
    McpToolNameError,
)
from base.models.view_models.mcp import McpServerConfig
from base.services.configuration.mixins import GuardrailSettingsMixin
from base.services.guardrails import GuardrailBlocked
from base.services.guardrails.injection import InjectionHeuristicGuardrail
from base.services.guardrails.pipeline import GuardrailPipeline
from base.services.mcp import McpClient
from base.services.tool_calling import Toolbox
from tests.conftest import MakeSettings

FIXTURE = Path(__file__).with_name("fixture_server.py")


@dataclass
class Servers:
    """A `SupportsMcpClientConfig` holding these servers, without composing settings."""

    mcp_servers: list[McpServerConfig] = field(default_factory=list)


class FixtureServer:
    """The fixture server as a subprocess on a port of its own, restartable in place."""

    def __init__(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = int(probe.getsockname()[1])
        self.url = f"http://127.0.0.1:{self.port}/mcp/"
        self._process: subprocess.Popen[bytes] | None = None

    async def astart(self) -> None:
        self._process = subprocess.Popen([sys.executable, str(FIXTURE), str(self.port)])
        for _ in range(200):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.1).close()
                return
            except OSError:
                await asyncio.sleep(0.05)
        raise RuntimeError("fixture server did not start")

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._process = None

    async def aauthorization_seen(self) -> list[str]:
        async with httpx2.AsyncClient() as http:
            seen = (await http.get(f"http://127.0.0.1:{self.port}/seen")).json()
        return [header for header in seen if header]


@pytest.fixture
async def server() -> AsyncIterator[FixtureServer]:
    fixture = FixtureServer()
    await fixture.astart()
    try:
        yield fixture
    finally:
        fixture.stop()


def config(url: str, name: str = "weather", credential: str = "") -> McpServerConfig:
    return McpServerConfig(
        name=name, transport=McpTransport.HTTP, url=url, credential=credential
    )


@pytest.fixture
async def client(server: FixtureServer) -> AsyncIterator[McpClient]:
    mcp = McpClient(config(server.url))
    await mcp.aconnect()
    try:
        yield mcp
    finally:
        await mcp.aclose()


def call(name: str, arguments: dict[str, object]) -> StoredToolCall:
    return StoredToolCall(
        id="call-1",
        function=StoredToolCallFunction(name=name, arguments=json.dumps(arguments)),
    )


# Tools as ours


async def test_each_tool_is_namespaced_by_its_server(client: McpClient) -> None:
    names = [tool.name for tool in await client.atools()]
    assert names == ["weather__forecast", "weather__refuse", "weather__inject"]


async def test_a_tool_keeps_the_schema_the_server_declared(client: McpClient) -> None:
    [forecast, *_] = await client.atools()
    assert forecast.json_schema == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }


async def test_two_servers_with_the_same_tool_do_not_collide(
    server: FixtureServer,
) -> None:
    first, second = McpClient(config(server.url)), McpClient(config(server.url, "maps"))
    await first.aconnect()
    await second.aconnect()
    try:
        tools = [*await first.atools(), *await second.atools()]
        assert {"weather__forecast", "maps__forecast"} <= set(Toolbox(*tools).names)
    finally:
        await first.aclose()
        await second.aclose()


async def test_a_call_reaches_the_server(client: McpClient) -> None:
    toolbox = Toolbox(*await client.atools())
    assert await toolbox.ainvoke(call("weather__forecast", {"city": "Cairo"})) == (
        "sunny in Cairo"
    )


async def test_arguments_are_checked_before_they_leave(client: McpClient) -> None:
    toolbox = Toolbox(*await client.atools())
    answer = await toolbox.ainvoke(call("weather__forecast", {}))
    assert answer.startswith("Error:") and "'city' is a required property" in answer


async def test_a_result_the_server_flags_as_an_error_reads_as_one(
    client: McpClient,
) -> None:
    toolbox = Toolbox(*await client.atools())
    assert await toolbox.ainvoke(call("weather__refuse", {})) == (
        "Error: the server refused"
    )


# Auth


async def test_a_configured_credential_is_sent_as_a_bearer(
    server: FixtureServer,
) -> None:
    mcp = McpClient(config(server.url), StaticTokenProvider("server-token"))
    await mcp.aconnect()
    try:
        await mcp.atools()
    finally:
        await mcp.aclose()
    seen = await server.aauthorization_seen()
    assert seen and set(seen) == {"Bearer server-token"}


async def test_no_credential_sends_no_authorization(
    client: McpClient, server: FixtureServer
) -> None:
    await client.atools()
    assert await server.aauthorization_seen() == []


# Untrusted results


async def test_a_hostile_result_is_stopped_at_the_tool_result_stage(
    client: McpClient, make_settings: MakeSettings
) -> None:
    """An MCP server nobody here reviewed is the reason the TOOL_RESULT stage exists."""
    make_settings(GuardrailSettingsMixin)
    guardrails = GuardrailPipeline(InjectionHeuristicGuardrail())
    toolbox = Toolbox(*await client.atools())
    answer = await toolbox.ainvoke(call("weather__inject", {}))
    with pytest.raises(GuardrailBlocked):
        await guardrails.aenforce(answer, GuardrailStage.TOOL_RESULT)


# Staying connected


async def test_a_server_that_restarts_is_reconnected_to(
    client: McpClient, server: FixtureServer
) -> None:
    toolbox = Toolbox(*await client.atools())
    server.stop()
    await server.astart()
    assert await toolbox.ainvoke(call("weather__forecast", {"city": "Doha"})) == (
        "sunny in Doha"
    )


async def test_a_server_that_stays_down_is_a_domain_error(
    client: McpClient, server: FixtureServer
) -> None:
    server.stop()
    with pytest.raises(McpConnectionError):
        await client.atools()


async def test_a_server_that_never_answers_is_a_timeout(
    client: McpClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK lets a timeout escape as a raw httpx2 error; it is ours by the time it leaves."""

    async def hang(*args: object, **kwargs: object) -> object:
        raise httpx2.ReadTimeout("no answer")

    session = client._session
    assert session is not None
    monkeypatch.setattr(session, "list_tools", hang)

    async def keep_session() -> None:
        client._session = session

    monkeypatch.setattr(client, "_arecover", keep_session)
    with pytest.raises(McpTimeoutError) as raised:
        await client.atools()
    assert raised.value.attempts == 2


async def test_an_unreachable_server_fails_to_connect_as_a_domain_error() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    mcp = McpClient(config(f"http://127.0.0.1:{port}/mcp/"))
    with pytest.raises(McpConnectionError):
        await mcp.aconnect()
    await mcp.aclose()


# Configuration


def test_a_tool_name_no_provider_accepts_is_refused() -> None:
    """MCP allows 128 characters and dots; every model provider stops at 64 and refuses dots."""
    mcp = McpClient(config("http://unused/"))
    for name in ("x" * 60, "read.file"):
        with pytest.raises(McpToolNameError):
            mcp._wrap(types.Tool(name=name, input_schema={"type": "object"}))


async def test_duplicate_server_names_are_refused() -> None:
    clients = McpClients(
        config=Servers([config("http://a/", "docs"), config("http://b/", "docs")])
    )
    with pytest.raises(McpConfigError, match="unique"):
        await clients.astartup(None)  # type: ignore[arg-type]


async def test_a_credential_that_was_not_provided_is_refused() -> None:
    clients = McpClients(
        config=Servers([config("http://a/", "docs", credential="docs")])
    )
    with pytest.raises(McpConfigError, match="not provided"):
        await clients.astartup(None)  # type: ignore[arg-type]


def test_the_toolbox_cannot_be_read_before_startup() -> None:
    with pytest.raises(McpConfigError, match="before startup"):
        _ = McpClients(config=Servers()).toolbox
