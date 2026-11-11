"""The MCP server capability under a real uvicorn server and a real MCP client."""

import uuid
from typing import Any

import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.shared.exceptions import MCPError

from base.models.view_models import JSONObject
from base.services.auth import TokenIssuer
from tests.unit.mcp.conftest import Forecast, token


async def session_call(
    url: str, bearer: str | None, name: str, args: JSONObject
) -> Any:
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else None
    async with (
        create_mcp_http_client(headers=headers) as http,
        streamable_http_client(url, http_client=http) as (read, write, *_),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        if name == "__list__":
            return await session.list_tools()
        return await session.call_tool(name, args)


async def test_tools_are_listed_with_the_toolbox_schema(served: str) -> None:
    listed = await session_call(served, token(), "__list__", {})
    schemas = {tool.name: tool.input_schema for tool in listed.tools}
    assert schemas["forecast"] == Forecast.model_json_schema()
    assert schemas["whoami"] == {"type": "object", "properties": {}}


async def test_a_call_runs_the_toolbox_tool(served: str) -> None:
    result = await session_call(
        served, token(), "forecast", {"city": "Cairo", "days": 3}
    )
    assert not result.is_error
    assert result.content[0].text == "Cairo for 3d"


async def test_arguments_the_schema_refuses_come_back_as_an_error(served: str) -> None:
    result = await session_call(served, token(), "forecast", {"days": 3})
    assert result.is_error
    assert "city" in result.content[0].text


async def test_a_tool_that_raises_is_reported_without_its_internals(
    served: str,
) -> None:
    """The calling model is told it failed; the exception text stays in this service's log."""
    result = await session_call(served, token(), "explode", {})
    assert result.is_error
    assert result.content[0].text == "explode failed: RuntimeError"


async def test_an_unknown_tool_is_a_protocol_error(served: str) -> None:
    with pytest.RaisesGroup(
        pytest.RaisesExc(MCPError, match="Unknown tool"), flatten_subgroups=True
    ):
        await session_call(served, token(), "no_such_tool", {})


INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}


async def status_of(url: str, authorization: str | None) -> int:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if authorization is not None:
        headers["Authorization"] = authorization
    async with httpx2.AsyncClient() as http:
        return (await http.post(url, json=INITIALIZE, headers=headers)).status_code


async def test_a_call_without_a_token_is_refused(served: str) -> None:
    assert await status_of(served, None) == 401


async def test_a_call_with_a_forged_token_is_refused(served: str) -> None:
    assert await status_of(served, "Bearer not-a-real-token") == 401


async def test_a_call_with_a_valid_token_is_accepted(served: str) -> None:
    assert await status_of(served, f"Bearer {token()}") == 200


async def test_a_call_runs_as_the_verified_caller_and_their_tenant(served: str) -> None:
    user = uuid.uuid4()
    bearer = TokenIssuer().issue_access({"user_id": str(user), "tid": "acme"})
    result = await session_call(served, bearer, "whoami", {})
    assert result.content[0].text == f"tenant=acme caller={user}"
