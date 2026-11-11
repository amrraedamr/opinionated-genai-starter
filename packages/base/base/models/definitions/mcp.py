"""How an MCP server is reached, and how its tools are named once they are ours."""

import re
from enum import StrEnum
from typing import Final

MCP_TOOL_SEPARATOR: Final = "__"
"""Between a server's name and its tool's, so two servers cannot collide in one toolbox."""

PROVIDER_TOOL_NAME: Final = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
"""What every model provider accepts; MCP also allows `.` and 128 characters, which they reject."""

MCP_SERVER_NAME: Final = re.compile(r"^[A-Za-z0-9-]{1,32}$")
"""No underscore, so the separator after it is unambiguous."""

MCP_DEFAULT_MOUNT_PATH: Final = "/mcp"
MCP_RECONNECT_ATTEMPTS: Final = 1

MCP_UNPUBLISHED_ISSUER: Final = "https://unpublished.invalid"
"""The SDK wires the bearer check only beside an issuer; without a resource URL it is never published or read."""

MCP_PRINCIPAL_CLAIM: Final = "principal"
"""Where the verified `Principal` rides on the SDK's access token, from bearer check to tool."""


class McpTransport(StrEnum):
    """How this service reaches a server."""

    HTTP = "http"
    """Streamable HTTP, for a server running elsewhere."""

    STDIO = "stdio"
    """A subprocess this service starts and talks to over its standard streams."""
