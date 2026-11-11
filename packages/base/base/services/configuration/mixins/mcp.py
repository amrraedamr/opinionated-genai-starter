from pydantic import BaseModel

from base.models.definitions.mcp import MCP_DEFAULT_MOUNT_PATH
from base.models.view_models.mcp import McpServerConfig


class McpClientSettingsMixin(BaseModel):
    """The MCP servers this service calls tools on."""

    mcp_servers: list[McpServerConfig] = []
    """JSON in the environment, e.g. `[{"name": "docs", "transport": "http", "url": "..."}]`."""


class McpServerSettingsMixin(BaseModel):
    """Whether this service offers its own tools over MCP, and where."""

    mcp_server_enabled: bool = False
    mcp_server_path: str = MCP_DEFAULT_MOUNT_PATH
