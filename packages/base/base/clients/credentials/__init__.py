"""Outbound credentials: how a service authenticates itself to a dependency.

The mirror image of `base.services.auth`, which authenticates an inbound
caller. The two stay separate because their lifecycles differ: inbound
verifies a token minted elsewhere, per request, on a user's behalf; outbound
mints and caches a token for the service itself.
"""

from collections.abc import AsyncGenerator, Generator

import httpx

from base.clients.credentials.protocol import OutboundAuth, TokenProvider
from base.models.view_models.tokens import AccessToken
from base.tools.tenancy import BearerContext


class CallerTokenMissing(RuntimeError):
    """A call meant to act as the caller has no live token of theirs to send."""


class StaticTokenProvider:
    """A fixed token, for local development and tests."""

    def __init__(self, token: AccessToken | str) -> None:
        self._token: AccessToken = AccessToken(str(token))

    async def aget_token(self) -> AccessToken:
        return self._token


class ForwardedTokenProvider:
    """The token the current caller presented, so a dependency sees them rather than this service."""

    async def aget_token(self) -> AccessToken:
        bearer = BearerContext.get()
        if bearer is None:
            raise CallerTokenMissing(
                "No caller token in context; forward it only from inside an authenticated request"
            )
        if bearer.expired:
            raise CallerTokenMissing("The caller's token has expired")
        return bearer.token


class BearerTokenAuth(httpx.Auth):
    """Sets `Authorization: Bearer` from a provider, per request.

    Asking per request rather than capturing once is what makes expiry
    invisible to callers: the provider's cache decides when to refresh.
    """

    def __init__(self, provider: TokenProvider) -> None:
        self._provider = provider

    @property
    def provider(self) -> TokenProvider:
        return self._provider

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        # Yielding unchanged would send the request with no credentials.
        raise RuntimeError(
            "BearerTokenAuth acquires tokens asynchronously and cannot "
            "authenticate a synchronous request. Use the async client, or the "
            "`a`-prefixed methods of the service holding it."
        )

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = await self._provider.aget_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


__all__ = [
    "AccessToken",
    "BearerTokenAuth",
    "CallerTokenMissing",
    "ForwardedTokenProvider",
    "OutboundAuth",
    "StaticTokenProvider",
    "TokenProvider",
]
