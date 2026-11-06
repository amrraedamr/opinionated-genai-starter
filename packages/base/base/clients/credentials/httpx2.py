"""Bearer auth bound to `httpx2`.

`httpx2` is a separate distribution from `httpx`, with its own `Auth` base
class that rejects `httpx`'s outright, so `BearerTokenAuth` cannot be handed to
a client built on it — the Anthropic SDK being the one in this library. This is
the same behaviour against the other library; both read from one
`TokenProvider`, so a token acquired for either is the token used by both.

Kept out of the package `__init__` because `httpx2` arrives only with the
`anthropic` extra, while `httpx` is a core dependency.
"""

from collections.abc import AsyncGenerator, Generator

from base.tools.errors import MissingExtraError

try:
    import httpx2
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "anthropic", "httpx2", feature="httpx2 bearer auth"
    ) from exc

from base.clients.credentials.protocol import TokenProvider


class Httpx2BearerTokenAuth(httpx2.Auth):
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
        self, request: httpx2.Request
    ) -> Generator[httpx2.Request, httpx2.Response, None]:
        # Yielding unchanged would send the request with no credentials.
        raise RuntimeError(
            "Httpx2BearerTokenAuth acquires tokens asynchronously and cannot "
            "authenticate a synchronous request. Use the async client, or the "
            "`a`-prefixed methods of the service holding it."
        )

    async def async_auth_flow(
        self, request: httpx2.Request
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        token = await self._provider.aget_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request
