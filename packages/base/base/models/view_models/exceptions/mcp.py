"""Domain errors the MCP client raises. No SDK type is named here.

Its own family, so a tool server going away is never mistaken for a model failure.
Smaller than the others: the SDK reports a rate limit and a server error as one
error with no status, so only a dropped connection and a timeout can be told apart.
"""

from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class McpError(ProviderError):
    """Base class for all MCP client errors."""


class McpConfigError(McpError, ProviderConfigError):
    """A server is misconfigured: a duplicate name, an unknown credential."""


class McpToolNameError(McpError):
    """A server offered a tool whose name no model provider accepts."""


class McpRetryExhausted(McpError, ProviderRetryExhausted):
    """The server stayed unavailable after reconnecting."""


class McpConnectionError(McpRetryExhausted):
    """The server could not be reached, or dropped and reconnecting did not help."""


class McpTimeoutError(McpRetryExhausted):
    """The server did not answer in time, after reconnecting."""
