"""Offer a service's own tools over MCP, as the same verified caller its routes see."""

import contextlib

from base.tools.errors import MissingExtraError

try:
    from fastapi import FastAPI
    from mcp import types
    from mcp.server.auth.middleware.auth_context import get_access_token
    from mcp.server.auth.provider import AccessToken as McpAccessToken
    from mcp.server.auth.settings import AuthSettings
    from mcp.server.context import ServerRequestContext
    from mcp.server.lowlevel import Server
    from mcp.shared.exceptions import MCPError
    from pydantic import AnyHttpUrl
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("mcp", "mcp", feature="McpServer") from exc

from base.models.definitions.health import ProbeStatus
from base.models.definitions.mcp import MCP_PRINCIPAL_CLAIM, MCP_UNPUBLISHED_ISSUER
from base.models.view_models import BaseModel
from base.models.view_models.auth import Principal
from base.models.view_models.exceptions import McpConfigError
from base.models.view_models.health import ProbeResult
from base.services.auth.tokens import TokenVerifier
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsMcpServerConfig
from base.services.logging.console import ConsoleLoggerService
from base.services.tool_calling import Tool, Toolbox, ToolError
from base.tools.errors import ErrorsException
from base.tools.serializers import serialize
from base.tools.tenancy import CallerContext, TenantContext


class _PrincipalVerifier:
    """The SDK's bearer check, answered by this service's own `TokenVerifier`."""

    async def verify_token(self, token: str) -> McpAccessToken | None:
        try:
            principal = TokenVerifier().verify(token)
        except ErrorsException:
            return None
        return McpAccessToken(
            token=token,
            client_id=principal.subject,
            scopes=sorted(principal.scopes),
            subject=principal.subject,
            claims={MCP_PRINCIPAL_CLAIM: principal.model_dump(mode="json")},
        )


class McpServer:
    """Mounts a toolbox at `mcp_server_path`, refusing any call without a valid ******

    Tools are listed and called straight from the toolbox, so a tool's schema and
    validation are the ones every agent already uses. Stateless, because the SDK ties
    a session to the token that opened it and a client forwarding each caller's own
    token sends a different one per call. The session manager is started here
    because a mounted app's own lifespan never runs under FastAPI.
    """

    name = "mcp_server"

    def __init__(
        self,
        toolbox: Toolbox,
        config: SupportsMcpServerConfig | None = None,
        server_name: str = "opinionated-genai-starter",
    ) -> None:
        self._toolbox = toolbox
        self._tools: dict[str, Tool[BaseModel]] = {t.name: t for t in toolbox.tools}
        self._config = config
        self._server = Server(
            server_name, on_list_tools=self._alist, on_call_tool=self._acall
        )
        self._sessions: contextlib.AsyncExitStack | None = None
        self._logger = ConsoleLoggerService()

    @property
    def settings(self) -> SupportsMcpServerConfig:
        return self._config or settings_for(SupportsMcpServerConfig)

    async def astartup(self, app: FastAPI) -> None:
        if not self.settings.mcp_server_enabled:
            return
        path = self.settings.mcp_server_path
        mounted = self._server.streamable_http_app(
            streamable_http_path="/",
            stateless_http=True,
            auth=AuthSettings(
                issuer_url=AnyHttpUrl(MCP_UNPUBLISHED_ISSUER), resource_server_url=None
            ),
            token_verifier=_PrincipalVerifier(),
            host="0.0.0.0",
        )
        app.mount(path, mounted)
        self._sessions = contextlib.AsyncExitStack()
        await self._sessions.enter_async_context(self._server.session_manager.run())
        self._logger.info(f"MCP: serving {len(self._toolbox)} tool(s) at {path}")

    async def ashutdown(self, app: FastAPI) -> None:
        del app
        if self._sessions is not None:
            await self._sessions.aclose()
            self._sessions = None

    async def aprobe(self) -> ProbeResult:
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.OK,
            detail=None if self.settings.mcp_server_enabled else "disabled",
        )

    async def _alist(
        self, ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        del ctx, params
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.json_schema,
                )
                for tool in self._toolbox.tools
            ]
        )

    async def _acall(
        self, ctx: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        """Run the tool as the verified caller, with their tenant, as a route would.

        A tool's own failure is returned flagged as an error, so the calling model
        can recover; an unknown tool is the protocol error the SDK reports for it.
        """
        del ctx
        principal = self._principal()
        tool = self._tools.get(params.name)
        if tool is None:
            raise MCPError(types.INVALID_PARAMS, f"Unknown tool: {params.name}")
        with (
            TenantContext.scope(principal.tenant_id),
            CallerContext.scope(str(principal.user_id or principal.subject)),
        ):
            try:
                text = await tool.ainvoke(serialize(params.arguments or {}))
            except ToolError as exc:
                return self._failed(str(exc))
            except Exception as exc:
                self._logger.exception(f"MCP tool {params.name!r} failed")
                return self._failed(f"{params.name} failed: {type(exc).__name__}")
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    @staticmethod
    def _failed(message: str) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=message)], is_error=True
        )

    @staticmethod
    def _principal() -> Principal:
        """The caller the bearer check verified; the route refuses before any call without one."""
        access = get_access_token()
        if access is None or not access.claims:
            raise McpConfigError("MCP call reached a tool without a verified caller")
        return Principal.model_validate(access.claims[MCP_PRINCIPAL_CLAIM])
