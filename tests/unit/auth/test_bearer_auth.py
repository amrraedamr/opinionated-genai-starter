"""Signing an outgoing request with a bearer token.

The point of asking the provider per request is that expiry stays invisible to
callers: a token captured once would be sent until it was refused. The
synchronous path has no way to await a token, so it refuses rather than
sending the request unauthenticated — pinned, because yielding unchanged
would be a silent 401 at best and an unauthenticated call at worst.
"""

from typing import Any

import httpx
import httpx2
import pytest

from base.clients.credentials import BearerTokenAuth, StaticTokenProvider
from base.clients.credentials.httpx2 import Httpx2BearerTokenAuth
from base.models.view_models.tokens import AccessToken


class CountingProvider:
    """A different token each time it is asked, so reuse is visible."""

    def __init__(self) -> None:
        self.calls = 0

    async def aget_token(self) -> AccessToken:
        self.calls += 1
        return AccessToken(f"token-{self.calls}")


async def _run_flow(auth: BearerTokenAuth, request: httpx.Request) -> httpx.Request:
    flow = auth.async_auth_flow(request)
    signed = await anext(flow)
    await flow.aclose()
    return signed


def test_the_provider_is_reachable_for_inspection() -> None:
    provider = StaticTokenProvider("token")
    assert BearerTokenAuth(provider).provider is provider
    assert Httpx2BearerTokenAuth(provider).provider is provider


async def test_the_authorization_header_carries_the_providers_token() -> None:
    auth = BearerTokenAuth(StaticTokenProvider("a-token"))
    request = httpx.Request("GET", "https://api.example.com/things")

    signed = await _run_flow(auth, request)
    assert signed.headers["Authorization"] == "Bearer a-token"


async def test_the_token_is_fetched_for_every_request_not_captured_once() -> None:
    provider = CountingProvider()
    auth = BearerTokenAuth(provider)

    first = await _run_flow(auth, httpx.Request("GET", "https://api.example.com/a"))
    second = await _run_flow(auth, httpx.Request("GET", "https://api.example.com/b"))

    assert first.headers["Authorization"] == "Bearer token-1"
    assert second.headers["Authorization"] == "Bearer token-2"
    assert provider.calls == 2


async def test_an_existing_authorization_header_is_replaced() -> None:
    auth = BearerTokenAuth(StaticTokenProvider("the-real-token"))
    request = httpx.Request(
        "GET",
        "https://api.example.com/things",
        headers={"Authorization": "Bearer stale"},
    )

    signed = await _run_flow(auth, request)
    assert signed.headers.get_list("Authorization") == ["Bearer the-real-token"]


async def test_the_request_reaches_the_transport_authenticated() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json={})

    provider = CountingProvider()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), auth=BearerTokenAuth(provider)
    ) as client:
        await client.get("https://api.example.com/a")
        await client.get("https://api.example.com/b")

    assert seen == ["Bearer token-1", "Bearer token-2"]


def test_the_synchronous_flow_refuses_rather_than_sending_unauthenticated() -> None:
    auth = BearerTokenAuth(StaticTokenProvider("a-token"))
    request = httpx.Request("GET", "https://api.example.com/things")

    with pytest.raises(RuntimeError) as caught:
        auth.sync_auth_flow(request)
    message = str(caught.value)
    assert "asynchronously" in message
    assert "async client" in message


def test_a_synchronous_client_fails_instead_of_calling_out() -> None:
    called: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(request)
        return httpx.Response(200, json={})

    auth = BearerTokenAuth(StaticTokenProvider("a-token"))
    with httpx.Client(transport=httpx.MockTransport(handler), auth=auth) as client:
        with pytest.raises(RuntimeError):
            client.get("https://api.example.com/things")

    assert called == []


async def test_the_httpx2_binding_signs_the_same_way() -> None:
    provider = CountingProvider()
    auth = Httpx2BearerTokenAuth(provider)
    request = httpx2.Request("GET", "https://api.example.com/things")

    flow = auth.async_auth_flow(request)
    signed = await anext(flow)
    await flow.aclose()

    assert signed.headers["Authorization"] == "Bearer token-1"
    assert provider.calls == 1


def test_the_httpx2_synchronous_flow_refuses_too() -> None:
    auth = Httpx2BearerTokenAuth(StaticTokenProvider("a-token"))
    request = httpx2.Request("GET", "https://api.example.com/things")

    with pytest.raises(RuntimeError, match="Httpx2BearerTokenAuth"):
        auth.sync_auth_flow(request)


async def test_a_provider_failure_surfaces_rather_than_sending_the_request() -> None:
    class FailingProvider:
        async def aget_token(self) -> AccessToken:
            raise RuntimeError("the authorization server is down")

    called: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=BearerTokenAuth(FailingProvider()),
    ) as client:
        with pytest.raises(RuntimeError, match="authorization server is down"):
            await client.get("https://api.example.com/things")

    assert called == []
