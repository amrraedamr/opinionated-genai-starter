"""The MCP servers chat_api calls tools on, each reached as the user who is chatting."""

from base.capabilities.mcp_clients import McpClients
from base.clients.credentials import ForwardedTokenProvider

CALLER_CREDENTIAL = "caller"
"""Name a server's `credential` with this in `MCP_SERVERS` to send the caller's own token."""

MCP = McpClients(credentials={CALLER_CREDENTIAL: ForwardedTokenProvider()})
