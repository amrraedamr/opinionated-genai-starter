"""The reference MCP service, run as it deploys, called by the library's own MCP client.

The service runs in a process and virtual environment of its own, configured
only through the environment, so this proves the shipped `app` serves MCP, not
that an app assembled inside the test does.
"""

import json
import os
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import jwt
import pytest

from base.clients.credentials import StaticTokenProvider
from base.models.definitions.mcp import McpTransport
from base.models.view_models.chat_history import StoredToolCall, StoredToolCallFunction
from base.models.view_models.exceptions import McpConnectionError
from base.models.view_models.mcp import McpServerConfig
from base.services.auth import TokenIssuer
from base.services.configuration.mixins import TokenIssuingSettingsMixin
from base.services.mcp import McpClient
from base.services.tool_calling import Toolbox
from tests.conftest import MakeSettings
from tests.unit.auth.conftest import _generate_rsa

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "services" / "mcp_server"
VENV = ROOT / ".venvs" / "mcp_server"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def keys(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    private_pem, public_pem = _generate_rsa()
    directory = tmp_path_factory.mktemp("mcp-keys")
    private_path, public_path = directory / "private.pem", directory / "public.pem"
    private_path.write_text(private_pem)
    public_path.write_text(public_pem)
    return private_path, public_path


@pytest.fixture(scope="module")
def service(keys: tuple[Path, Path]) -> Iterator[str]:
    """The mcp_server package serving on a free port, verifying with `keys`."""
    _, public_path = keys
    subprocess.run(
        ["uv", "sync", "-q", "--package", "mcp_server"],
        cwd=ROOT,
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(VENV)},
        check=True,
    )
    port = free_port()
    environment = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(VENV),
        "ENVIRONMENT": "local",
        "AUTH_PUBLIC_KEY_PATH": str(public_path),
        "TENANCY_ENABLED": "true",
    }
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "--no-sync",
            "--package",
            "mcp_server",
            "uvicorn",
            "mcp_server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=SERVICE,
        env=environment,
    )
    try:
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError(f"mcp_server exited with {process.returncode}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("mcp_server did not start listening")
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture
def issuer(make_settings: MakeSettings, keys: tuple[Path, Path]) -> TokenIssuer:
    """Issues tokens with the private half of the key the service verifies with."""
    private_path, public_path = keys
    make_settings(
        TokenIssuingSettingsMixin,
        auth_private_key_path=str(private_path),
        auth_public_key_path=str(public_path),
    )
    return TokenIssuer()


def call(name: str, arguments: dict[str, object]) -> StoredToolCall:
    return StoredToolCall(
        id="c",
        function=StoredToolCallFunction(name=name, arguments=json.dumps(arguments)),
    )


async def connected(url: str, bearer: str | None) -> McpClient:
    client = McpClient(
        McpServerConfig(name="ref", transport=McpTransport.HTTP, url=url),
        StaticTokenProvider(bearer) if bearer else None,
    )
    await client.aconnect()
    return client


async def test_the_service_offers_its_tools_namespaced(
    service: str, issuer: TokenIssuer
) -> None:
    client = await connected(
        service, issuer.issue_access({"user_id": str(uuid.uuid4())})
    )
    try:
        names = Toolbox(*await client.atools()).names
    finally:
        await client.aclose()
    assert names == ["ref__add", "ref__utc_now", "ref__whoami", "ref__word_count"]


async def test_a_tool_runs_on_the_service(service: str, issuer: TokenIssuer) -> None:
    client = await connected(
        service, issuer.issue_access({"user_id": str(uuid.uuid4())})
    )
    try:
        toolbox = Toolbox(*await client.atools())
        assert await toolbox.ainvoke(call("ref__add", {"a": 2, "b": 3})) == "5.0"
        assert await toolbox.ainvoke(call("ref__word_count", {"text": "a b c"})) == "3"
    finally:
        await client.aclose()


async def test_the_service_runs_a_call_as_the_tokens_tenant(
    service: str, issuer: TokenIssuer
) -> None:
    user = uuid.uuid4()
    client = await connected(
        service, issuer.issue_access({"user_id": str(user), "tid": "acme"})
    )
    try:
        toolbox = Toolbox(*await client.atools())
        assert await toolbox.ainvoke(call("ref__whoami", {})) == (
            f"caller={user} tenant=acme"
        )
    finally:
        await client.aclose()


async def test_the_service_refuses_a_client_without_a_token(service: str) -> None:
    client = McpClient(
        McpServerConfig(name="ref", transport=McpTransport.HTTP, url=service)
    )
    with pytest.raises(McpConnectionError):
        await client.aconnect()
    await client.aclose()


async def test_the_service_refuses_a_token_signed_with_another_key(
    service: str,
) -> None:
    """A real signed token, but not one this service can verify."""
    other_private, _ = _generate_rsa()
    forged = jwt.encode(
        {"user_id": str(uuid.uuid4()), "type": "access"}, other_private, "RS256"
    )
    with pytest.raises(McpConnectionError):
        await connected(service, forged)
