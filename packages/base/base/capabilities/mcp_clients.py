"""Hold a connection to every configured MCP server, and offer their tools as one toolbox."""

from collections.abc import Mapping

from base.tools.errors import MissingExtraError

try:
    from fastapi import FastAPI
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("server", "fastapi", feature="McpClients") from exc

from base.clients.credentials.protocol import TokenProvider
from base.models.definitions.health import ProbeStatus
from base.models.view_models import BaseModel
from base.models.view_models.exceptions import McpConfigError, McpError
from base.models.view_models.health import ProbeResult
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsMcpClientConfig
from base.services.logging.console import ConsoleLoggerService
from base.services.mcp import McpClient
from base.services.tool_calling import Tool, Toolbox


class McpClients:
    """Connects to each server on first use, closes in `ashutdown`, degrades `/readyz` while one is down.

    A server's `credential` names one of `credentials`, so a token never sits in settings::

        mcp = McpClients(credentials={"docs": OutboundCredentials().provider})
        app = create_app(capabilities=[mcp, ...])
        runner = ToolRunner(await mcp.atoolbox())
    """

    name = "mcp_clients"

    def __init__(
        self,
        config: SupportsMcpClientConfig | None = None,
        credentials: Mapping[str, TokenProvider] | None = None,
    ) -> None:
        self._config = config
        self._credentials = dict(credentials or {})
        self._clients: list[McpClient] | None = None
        self._logger = ConsoleLoggerService()

    async def atoolbox(self) -> Toolbox:
        """Every reachable server's tools, listed as the current caller; a down server is skipped."""
        if self._clients is None:
            raise McpConfigError(
                "McpClients.atoolbox was called before startup; call it inside a request"
            )
        tools: list[Tool[BaseModel]] = []
        for client in self._clients:
            try:
                tools.extend(await client.atools())
            except McpError as exc:
                self._logger.warning(f"MCP server {client.name!r} unavailable: {exc}")
        return Toolbox(*tools)

    async def astartup(self, app: FastAPI) -> None:
        """Check the configuration; each server is first reached by a caller's request."""
        del app
        self._clients = self._build()
        self._logger.info(f"MCP: {len(self._clients)} server(s) configured")

    async def ashutdown(self, app: FastAPI) -> None:
        del app
        for client in self._clients or []:
            await client.aclose()

    async def aprobe(self) -> ProbeResult:
        """Degraded while a server's last call failed; chat still answers without its tools."""
        down = [client.name for client in self._clients or [] if client.failing]
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.DEGRADED if down else ProbeStatus.OK,
            detail=f"last call failed: {', '.join(down)}" if down else None,
        )

    def _build(self) -> list[McpClient]:
        """One client per configured server, refusing duplicates and unknown credentials."""
        servers = (self._config or settings_for(SupportsMcpClientConfig)).mcp_servers
        names = [server.name for server in servers]
        if duplicates := sorted({n for n in names if names.count(n) > 1}):
            raise McpConfigError(f"MCP server names must be unique: {duplicates}")
        clients: list[McpClient] = []
        for server in servers:
            if server.credential and server.credential not in self._credentials:
                raise McpConfigError(
                    f"MCP server {server.name!r} names credential "
                    f"{server.credential!r}, which was not provided"
                )
            provider = self._credentials.get(server.credential)
            clients.append(McpClient(server, provider))
        return clients
