"""One MCP server this service connects to, as settings describe it."""

from typing import Self

from pydantic import model_validator

from base.models.definitions.mcp import MCP_SERVER_NAME, McpTransport
from base.models.view_models import BaseModel


class McpServerConfig(BaseModel):
    """Where a server is and how to reach it; its token comes from a named provider, not here."""

    name: str
    """Prefixes each of its tools as `name__tool`."""

    transport: McpTransport
    url: str = ""
    command: str = ""
    args: list[str] = []
    credential: str = ""
    """Names the `TokenProvider` handed to the capability; empty sends no token."""

    @model_validator(mode="after")
    def _reachable(self) -> Self:
        """Refuse at startup a server that names nothing to connect to."""
        if not MCP_SERVER_NAME.match(self.name):
            raise ValueError(
                f"MCP server name {self.name!r} must be 1-32 letters, digits or '-'"
            )
        if self.transport == McpTransport.HTTP and not self.url:
            raise ValueError(f"MCP server {self.name!r} uses http and needs a url")
        if self.transport == McpTransport.STDIO and not self.command:
            raise ValueError(f"MCP server {self.name!r} uses stdio and needs a command")
        return self
