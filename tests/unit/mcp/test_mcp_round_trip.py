"""The library's own MCP client calling the library's own MCP server: each proves the other."""

import uuid
from collections.abc import AsyncIterator

import pytest

from base.clients.credentials import StaticTokenProvider
from base.models.definitions.mcp import McpTransport
from base.models.view_models.exceptions import McpConnectionError
from base.models.view_models.mcp import McpServerConfig
from base.services.auth import TokenIssuer
from base.services.mcp import McpClient
from base.services.tool_calling import Toolbox
from tests.unit.mcp.test_mcp_client import call


@pytest.fixture
async def remote(served: str) -> AsyncIterator[Toolbox]:
    user = uuid.uuid4()
    bearer = TokenIssuer().issue_access({"user_id": str(user), "tid": "acme"})
    client = McpClient(
        McpServerConfig(name="self", transport=McpTransport.HTTP, url=served),
        StaticTokenProvider(bearer),
    )
    await client.aconnect()
    try:
        yield Toolbox(*await client.atools())
    finally:
        await client.aclose()


async def test_a_served_tool_arrives_namespaced(remote: Toolbox) -> None:
    assert {"self__forecast", "self__whoami", "self__explode"} <= set(remote.names)


async def test_a_pydantic_tool_round_trips(remote: Toolbox) -> None:
    answer = await remote.ainvoke(call("self__forecast", {"city": "Cairo", "days": 2}))
    assert answer == "Cairo for 2d"


async def test_the_server_runs_the_call_as_the_tokens_tenant(remote: Toolbox) -> None:
    assert (await remote.ainvoke(call("self__whoami", {}))).startswith("tenant=acme ")


async def test_a_failure_on_the_server_reads_as_an_error_here(remote: Toolbox) -> None:
    assert await remote.ainvoke(call("self__explode", {})) == (
        "Error: explode failed: RuntimeError"
    )


async def test_the_client_without_a_token_cannot_connect(served: str) -> None:
    client = McpClient(
        McpServerConfig(name="self", transport=McpTransport.HTTP, url=served)
    )
    with pytest.raises(McpConnectionError):
        await client.aconnect()
    await client.aclose()
