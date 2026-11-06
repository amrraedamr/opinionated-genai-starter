"""What outbound credentials have to satisfy, named without binding a client.

Separate from the package `__init__` because that module binds `httpx` to
implement them, and these two name no transport at all.
"""

from typing import Any, Protocol, runtime_checkable

from base.models.view_models.tokens import AccessToken


@runtime_checkable
class TokenProvider(Protocol):
    """Hands out a currently valid outbound token, refreshing as needed."""

    async def aget_token(self) -> AccessToken: ...


@runtime_checkable
class OutboundAuth(Protocol):
    """Anything that can sign an outgoing request.

    Structural on purpose. The OpenAI SDK ships on `httpx` and the Anthropic
    SDK on `httpx2`, whose `Auth` types are unrelated and reject each other, so
    naming either here would tie this module to one provider's dependency.
    """

    def sync_auth_flow(self, request: Any) -> Any: ...

    def async_auth_flow(self, request: Any) -> Any: ...


__all__ = ["OutboundAuth", "TokenProvider"]
