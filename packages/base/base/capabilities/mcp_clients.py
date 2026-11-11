"""Hold a connection to every configured MCP server, and offer their tools as one toolbox."""

from collections.abc import Mapping

from base.tools.errors import MissingExtraError

try:
    from fastapi import FastAPI
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("server", "fastapi", feature="McpClients") from exc

from base.clients.credentials.protocol import TokenProvider
from base.models.definitions.health import ProbeStatus
from base.models.view_models.exceptions import McpConfigError, McpError
from base.models.view_models.health import ProbeResult
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsMcpClientConfig
from base.services.logging.console import ConsoleLoggerService
from base.services.mcp import McpClient
from base.services.tool_calling import Toolbox


class McpClients:
    """Connects in `astartup`, closes in `ashutdown`, and fails `/readyz` while a server is down.

    A server's `credential` names one of `credentials`, so a token never sits in settings::

        mcp = McpClients(credentials={"docs": OutboundCredentials().provider})
        app = create_app(capabilities=[mcp, ...])
        runner = ToolRunner(mcp.toolbox)
    """

    name = "mcp_clients"

    def __init__(
        self,
        config: SupportsMcpClientConfig | None = None,
        credentials: Mapping[str, TokenProvider] | None = None,
    ) -> None:
        self._config = config
        self._credentials = dict(credentials or {})
        self._clients: list[McpClient] = []
        self._toolbox: Toolbox | None = None
        self._logger = ConsoleLoggerService()

    @property
    def toolbox(self) -> Toolbox:
        """Every connected server's tools, namespaced so none collide; read it per request."""
        if self._toolbox is None:
            raise McpConfigError(
                "McpClients.toolbox was read before startup; read it inside a request"
            )
        return self._toolbox

    async def astartup(self, app: FastAPI) -> None:
        del app
        self._clients = self._build()
        for client in self._clients:
            await client.aconnect()
        tools = [tool for client in self._clients for tool in await client.atools()]
        self._toolbox = Toolbox(*tools)
        self._logger.info(
            f"MCP: {len(self._clients)} server(s), {len(tools)} tool(s) available"
        )

    async def ashutdown(self, app: FastAPI) -> None:
        del app
        for client in self._clients:
            await client.aclose()

    async def aprobe(self) -> ProbeResult:
        """Ready only while every server answers; a tool list is the cheapest real request."""
        down: list[str] = []
        for client in self._clients:
            try:
                await client.atools()
            except McpError:
                down.append(client.name)
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.FAILED if down else ProbeStatus.OK,
            detail=f"unreachable: {', '.join(down)}" if down else None,
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
