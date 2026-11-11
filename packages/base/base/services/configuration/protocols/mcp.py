from typing import Protocol, runtime_checkable

from base.models.view_models.mcp import McpServerConfig


@runtime_checkable
class SupportsMcpClientConfig(Protocol):
    """Satisfied by `McpClientSettingsMixin`."""

    mcp_servers: list[McpServerConfig]


@runtime_checkable
class SupportsMcpServerConfig(Protocol):
    """Satisfied by `McpServerSettingsMixin`."""

    mcp_server_enabled: bool
    mcp_server_path: str
