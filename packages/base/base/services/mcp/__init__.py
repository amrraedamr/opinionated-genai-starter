"""A held connection to one MCP server, and its tools as ours."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable

from base.tools.errors import MissingExtraError

try:
    import httpx2
    from mcp import ClientSession, types
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import (
        TransportStreams,
        create_mcp_http_client,
        streamable_http_client,
    )
    from mcp.shared.exceptions import MCPError
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("mcp", "mcp", feature="McpClient") from exc

from base.clients.credentials.httpx2 import Httpx2BearerTokenAuth
from base.clients.credentials.protocol import TokenProvider
from base.models.definitions.mcp import (
    MCP_RECONNECT_ATTEMPTS,
    MCP_TOOL_SEPARATOR,
    PROVIDER_TOOL_NAME,
    McpTransport,
)
from base.models.view_models import BaseModel, JSONObject
from base.models.view_models.exceptions import (
    McpConnectionError,
    McpRetryExhausted,
    McpTimeoutError,
    McpToolNameError,
)
from base.models.view_models.mcp import McpServerConfig
from base.services.logging.console import ConsoleLoggerService
from base.services.tool_calling import Tool, ToolError


class McpClient:
    """Holds one session open and reconnects when the server drops.

    Async-only: the MCP SDK has no synchronous client. The session lives in a task
    of its own because the SDK's cancel scopes must close in the task that opened them.
    """

    def __init__(
        self, config: McpServerConfig, provider: TokenProvider | None = None
    ) -> None:
        self._config = config
        self._provider = provider
        self._logger = ConsoleLoggerService()
        self._session: ClientSession | None = None
        self._owner: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._reconnect = asyncio.Lock()
        self._failing = False

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def failing(self) -> bool:
        """Whether the last call failed; readiness has no caller token to try one itself."""
        return self._failing

    @property
    def connected(self) -> bool:
        return (
            self._session is not None
            and self._owner is not None
            and not self._owner.done()
        )

    async def aconnect(self) -> None:
        """Open the session, raising `McpConnectionError` when the server cannot be reached."""
        self._stop = asyncio.Event()
        ready = asyncio.Event()
        owner = asyncio.create_task(self._ahold(ready))
        self._owner = owner
        waiting = asyncio.ensure_future(ready.wait())
        done, _ = await asyncio.wait(
            {owner, waiting}, return_when=asyncio.FIRST_COMPLETED
        )
        if owner not in done:
            return
        waiting.cancel()
        self._session = None
        raise self._unavailable("could not be reached", owner.exception(), attempts=1)

    async def aclose(self) -> None:
        """Close the session. Safe to call when it never opened."""
        self._stop.set()
        if self._owner is not None:
            with contextlib.suppress(BaseException):
                await self._owner
        self._session = None
        self._owner = None

    async def atools(self) -> list[Tool[BaseModel]]:
        """The server's tools, each named `server__tool` and calling back through this client."""
        listed = await self._acall(lambda session: session.list_tools())
        return [self._wrap(tool) for tool in listed.tools]

    async def _ahold(self, ready: asyncio.Event) -> None:
        """Own the session for its whole life: open, signal ready, wait to be stopped."""
        async with (
            self._streams() as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            self._session = session
            ready.set()
            await self._stop.wait()

    @contextlib.asynccontextmanager
    async def _streams(self) -> AsyncIterator[TransportStreams]:
        """The transport's two streams, over HTTP or a subprocess."""
        if self._config.transport == McpTransport.STDIO:
            parameters = StdioServerParameters(
                command=self._config.command, args=self._config.args
            )
            async with stdio_client(parameters) as streams:
                yield streams
            return
        auth = Httpx2BearerTokenAuth(self._provider) if self._provider else None
        async with (
            create_mcp_http_client(auth=auth) as http,
            streamable_http_client(self._config.url, http_client=http) as streams,
        ):
            yield streams

    async def _acall[T](self, request: Callable[[ClientSession], Awaitable[T]]) -> T:
        """Run `request` on the session, connecting on first use and recording the outcome."""
        await self._apreflight()
        try:
            result = await self._aattempt(request)
        except McpRetryExhausted:
            self._failing = True
            raise
        self._failing = False
        return result

    async def _apreflight(self) -> None:
        """Get the token before the transport does: a provider raising in there ends the shared session."""
        if self._provider is None:
            return
        try:
            await self._provider.aget_token()
        except Exception as exc:
            raise McpConnectionError(
                f"MCP server {self.name!r} has no token to send: {exc}",
                attempts=1,
                provider_error=exc,
            ) from exc

    async def _aattempt[T](self, request: Callable[[ClientSession], Awaitable[T]]) -> T:
        """Run `request` on the session, reconnecting once if the server has dropped."""
        attempts = MCP_RECONNECT_ATTEMPTS + 1
        for attempt in range(1, attempts + 1):
            if not self.connected:
                await self._arecover()
            session = self._session
            if session is None:  # pragma: no cover - aconnect raises instead
                raise self._unavailable("has no session", None, attempts=attempt)
            try:
                return await request(session)
            except (MCPError, httpx2.TransportError) as exc:
                if attempt == attempts:
                    raise self._unavailable("is unavailable", exc, attempts) from exc
                self._logger.warning(f"MCP server {self.name!r} dropped; reconnecting")
                self._session = None
        raise AssertionError("unreachable")  # pragma: no cover

    def _unavailable(
        self, reason: str, cause: BaseException | None, attempts: int
    ) -> McpConnectionError | McpTimeoutError:
        """The domain error for a server that cannot be used, a timeout told apart."""
        error = (
            McpTimeoutError
            if isinstance(cause, httpx2.TimeoutException)
            else McpConnectionError
        )
        return error(
            f"MCP server {self.name!r} {reason}: {cause}",
            attempts=attempts,
            provider_error=cause if isinstance(cause, Exception) else None,
        )

    async def _arecover(self) -> None:
        """Replace a dead session; concurrent callers share the one reconnect."""
        stale = self._owner
        async with self._reconnect:
            if self._owner is not stale and self.connected:
                return
            await self.aclose()
            await self.aconnect()

    def _wrap(self, tool: types.Tool) -> Tool[BaseModel]:
        """One server tool as ours: namespaced, with its own schema, calling back here."""
        name = f"{self.name}{MCP_TOOL_SEPARATOR}{tool.name}"
        if not PROVIDER_TOOL_NAME.match(name):
            raise McpToolNameError(
                f"MCP tool {name!r} from {self.name!r} is not a name model providers accept"
            )
        remote = tool.name

        async def handler(arguments: JSONObject) -> str:
            result = await self._acall(
                lambda session: session.call_tool(remote, dict(arguments))
            )
            if not isinstance(result, types.CallToolResult):
                raise ToolError(f"{name} asked for input this client cannot give")
            text = self._render(result)
            if result.is_error:
                raise ToolError(text or f"{name} reported an error")
            return text

        return Tool.from_json_schema(
            name=name,
            description=tool.description or "",
            schema=dict(tool.input_schema),
            handler=handler,
        )

    @staticmethod
    def _render(result: types.CallToolResult) -> str:
        """A result as the text a model reads; non-text parts are named, not inlined."""
        return "\n".join(
            block.text
            if isinstance(block, types.TextContent)
            else f"[{block.type} content omitted]"
            for block in result.content
        )


__all__ = ["McpClient"]
