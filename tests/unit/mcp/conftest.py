"""What the MCP tests share: signing keys, and this library's MCP server running for real.

The served app runs in-process. That is safe only because nothing here lets a session
outlive the server; the client's drop and reconnect tests use `fixture_server` instead.
"""

import asyncio
import socket
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from base.app import create_app
from base.capabilities.mcp_server import McpServer
from base.models.view_models import BaseModel, JSONObject
from base.services.auth import TokenIssuer
from base.services.configuration.mixins import (
    McpServerSettingsMixin,
    ServerSettingsMixin,
    TenancySettingsMixin,
    TokenIssuingSettingsMixin,
)
from base.services.tool_calling import Tool, Toolbox
from base.tools.tenancy import CallerContext, TenantContext
from tests.conftest import MakeSettings
from tests.unit.auth.conftest import _generate_rsa

MOUNT = "/mcp"


class Forecast(BaseModel):
    city: str
    days: int = 1


async def forecast(args: Forecast) -> str:
    return f"{args.city} for {args.days}d"


async def whoami(arguments: JSONObject) -> str:
    del arguments
    return f"tenant={TenantContext.get()} caller={CallerContext.get()}"


async def explode(arguments: JSONObject) -> str:
    del arguments
    raise RuntimeError("database password is hunter2")


def served_toolbox() -> Toolbox:
    """A model-backed tool, a schema-backed one, and one that always raises."""
    empty: JSONObject = {"type": "object", "properties": {}}
    return Toolbox(
        Tool(
            name="forecast",
            description="Forecast.",
            parameters=Forecast,
            handler=forecast,
        ),
        Tool.from_json_schema(
            name="whoami",
            description="Who the call ran as.",
            schema=empty,
            handler=whoami,
        ),
        Tool.from_json_schema(
            name="explode", description="Always raises.", schema=empty, handler=explode
        ),
    )


def token(**claims: Any) -> str:
    """A genuine signed access token for a fresh user, with any extra claims."""
    return TokenIssuer().issue_access({"user_id": str(uuid.uuid4()), **claims})


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="session")
def rsa_pem() -> tuple[str, str]:
    return _generate_rsa()


@pytest.fixture
def key_paths(rsa_pem: tuple[str, str], tmp_path: Path) -> tuple[str, str]:
    """The session's RSA pair, written where settings can point at it."""
    private_pem, public_pem = rsa_pem
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_text(private_pem)
    public_path.write_text(public_pem)
    return str(private_path), str(public_path)


@pytest.fixture
async def served(
    make_settings: MakeSettings, key_paths: tuple[str, str]
) -> AsyncIterator[str]:
    """The MCP endpoint of an app serving `served_toolbox`, under a real uvicorn server."""
    private_path, public_path = key_paths
    settings = make_settings(
        ServerSettingsMixin,
        TokenIssuingSettingsMixin,
        TenancySettingsMixin,
        McpServerSettingsMixin,
        service_name="mcp_test",
        auth_private_key_path=private_path,
        auth_public_key_path=public_path,
        tenancy_enabled=True,
        mcp_server_enabled=True,
        mcp_server_path=MOUNT,
    )
    app = create_app(settings=settings, capabilities=[McpServer(served_toolbox())])
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}{MOUNT}/"
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, timeout=10)
