"""Outbound credentials: what a service presents to a protected dependency.

`TokenProvider` and `OutboundAuth` are structural on purpose — the OpenAI SDK
ships on `httpx` and the Anthropic SDK on `httpx2`, whose `Auth` types reject
each other. Pins that both protocols stay runtime-checkable, that the shipped
implementations satisfy them, and that the package re-export is the same
object as the protocol module's.
"""

from typing import Any, Protocol

import httpx
import httpx2

import base.clients.credentials as credentials_package
from base.clients.credentials import (
    BearerTokenAuth,
    OutboundAuth,
    StaticTokenProvider,
    TokenProvider,
)
from base.clients.credentials import protocol as protocol_module
from base.clients.credentials.httpx2 import Httpx2BearerTokenAuth
from base.clients.credentials.oauth2 import OAuth2ClientCredentials
from base.models.view_models.tokens import AccessToken


def test_the_package_re_exports_the_protocol_modules_own_objects() -> None:
    assert TokenProvider is protocol_module.TokenProvider
    assert OutboundAuth is protocol_module.OutboundAuth


def test_the_protocol_module_names_no_transport() -> None:
    """Importable by a service that installs neither httpx client."""
    assert protocol_module.__all__ == ["OutboundAuth", "TokenProvider"]


def test_both_protocols_are_runtime_checkable() -> None:
    assert TokenProvider._is_runtime_protocol is True
    assert OutboundAuth._is_runtime_protocol is True
    assert Protocol in TokenProvider.__bases__
    assert Protocol in OutboundAuth.__bases__


def test_a_static_token_provider_is_a_token_provider() -> None:
    assert isinstance(StaticTokenProvider("token"), TokenProvider)


def test_a_client_credentials_grant_is_a_token_provider() -> None:
    provider = OAuth2ClientCredentials(
        token_url="https://idp.example.com/token",
        client_id="id",
        client_secret="secret",
    )
    assert isinstance(provider, TokenProvider)


def test_an_object_with_no_aget_token_is_not_a_token_provider() -> None:
    class SynchronousOnly:
        def get_token(self) -> str:
            return "token"

    assert not isinstance(object(), TokenProvider)
    assert not isinstance(SynchronousOnly(), TokenProvider)


def test_bearer_token_auth_is_an_outbound_auth() -> None:
    assert isinstance(BearerTokenAuth(StaticTokenProvider("token")), OutboundAuth)


def test_the_httpx2_binding_is_an_outbound_auth_too() -> None:
    """Both halves read one provider, so a token acquired for either serves both."""
    assert isinstance(Httpx2BearerTokenAuth(StaticTokenProvider("token")), OutboundAuth)


def test_the_two_bindings_are_each_their_own_libraries_auth_type() -> None:
    provider = StaticTokenProvider("token")
    assert isinstance(BearerTokenAuth(provider), httpx.Auth)
    assert isinstance(Httpx2BearerTokenAuth(provider), httpx2.Auth)
    assert not isinstance(BearerTokenAuth(provider), httpx2.Auth)


def test_an_object_without_both_flows_is_not_an_outbound_auth() -> None:
    class HalfAnAuth:
        def sync_auth_flow(self, request: Any) -> Any: ...

    assert not isinstance(HalfAnAuth(), OutboundAuth)


async def test_a_static_token_provider_hands_back_the_token_it_was_given() -> None:
    provider = StaticTokenProvider("a-development-token")
    assert await provider.aget_token() == "a-development-token"
    assert await provider.aget_token() == "a-development-token"


async def test_a_static_token_provider_accepts_an_access_token_new_type() -> None:
    provider = StaticTokenProvider(AccessToken("a-development-token"))
    assert await provider.aget_token() == "a-development-token"


def test_the_package_exports_what_a_consumer_needs() -> None:
    assert sorted(credentials_package.__all__) == [
        "AccessToken",
        "BearerTokenAuth",
        "CallerTokenMissing",
        "ForwardedTokenProvider",
        "OutboundAuth",
        "StaticTokenProvider",
        "TokenProvider",
    ]
    for name in credentials_package.__all__:
        assert hasattr(credentials_package, name)
