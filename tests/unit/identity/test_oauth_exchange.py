"""The opaque code the SPA trades for a token pair.

It is a signed payload rather than a stored row, so what stops it being
anything else this service signs is the `type` claim. Every token in the
system is signed by the same key, which is exactly why that check matters.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from jwt import encode

from base.models.definitions.errors import AuthErrors, DatabaseErrors
from base.models.definitions.tokens import TokenType
from base.services.auth import TokenIssuer, TokenVerifier
from base.tools.errors import ErrorsException
from identity.models.definitions.auth import SignedPayloadType
from identity.models.orm.user import UserDTO
from identity.services.auth.oauth import OAuthService
from identity.stores.user import UserStore


@pytest.fixture
def service(identity_settings: Any) -> OAuthService:
    return OAuthService()


async def a_user(email: str = "person@example.com") -> UserDTO:
    return await UserStore().aregister(UserDTO(email=email, password_hash=None))


def code_for(user_id: str, ttl_seconds: int = 60) -> str:
    return TokenIssuer().sign_payload(
        {"type": SignedPayloadType.OAUTH_CODE, "user_id": user_id},
        ttl_seconds=ttl_seconds,
    )


# Spending a code


async def test_a_code_is_traded_for_a_pair_that_names_its_user(
    service: OAuthService,
) -> None:
    user = await a_user()
    pair = await service.aexchange_code(code_for(str(user.id)))
    claims = TokenVerifier().decode_local(
        pair.access_token, expected_type=TokenType.ACCESS
    )
    assert claims["user_id"] == str(user.id)
    assert claims["email"] == "person@example.com"


async def test_the_pair_a_code_yields_carries_both_tokens(
    service: OAuthService,
) -> None:
    user = await a_user()
    pair = await service.aexchange_code(code_for(str(user.id)))
    refresh = TokenVerifier().decode_local(
        pair.refresh_token, expected_type=TokenType.REFRESH
    )
    assert refresh["user_id"] == str(user.id)


async def test_a_code_naming_a_user_who_is_gone_is_a_record_not_found(
    service: OAuthService,
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(code_for(str(uuid.uuid4())))
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND


async def test_a_code_for_a_deleted_user_is_a_record_not_found(
    service: OAuthService,
) -> None:
    user = await a_user()
    code = code_for(str(user.id))
    await UserStore().adelete(user.id)
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(code)
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND


# Codes that are not codes


async def test_a_code_this_service_never_signed_is_refused(
    service: OAuthService,
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code("not-a-token")
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


async def test_an_empty_code_is_refused(service: OAuthService) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code("")
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


async def test_a_code_signed_by_another_key_is_refused(
    service: OAuthService,
) -> None:
    forged = encode(
        {"type": SignedPayloadType.OAUTH_CODE, "user_id": str(uuid.uuid4())},
        key="a key this service has never held",
        algorithm="HS256",
    )
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(forged)
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


async def test_a_tampered_code_is_refused(service: OAuthService) -> None:
    user = await a_user()
    code = code_for(str(user.id))
    head, payload, signature = code.split(".")
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(f"{head}.{payload}.{signature[::-1]}")
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


async def test_an_expired_code_is_refused(service: OAuthService) -> None:
    user = await a_user()
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(code_for(str(user.id), ttl_seconds=-10))
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


async def test_an_access_token_is_not_an_exchange_code(
    service: OAuthService,
) -> None:
    """Same key, same signature, different purpose."""
    user = await a_user()
    pair = TokenIssuer().issue_pair({"user_id": str(user.id), "email": user.email})
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(pair.access_token)
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


async def test_a_refresh_token_is_not_an_exchange_code(
    service: OAuthService,
) -> None:
    user = await a_user()
    pair = TokenIssuer().issue_pair({"user_id": str(user.id), "email": user.email})
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(pair.refresh_token)
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


async def test_a_signed_payload_of_no_declared_type_is_refused(
    service: OAuthService,
) -> None:
    user = await a_user()
    payload = TokenIssuer().sign_payload({"user_id": str(user.id)}, ttl_seconds=60)
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(payload)
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


async def test_a_code_naming_nobody_is_refused(service: OAuthService) -> None:
    code = TokenIssuer().sign_payload(
        {"type": SignedPayloadType.OAUTH_CODE}, ttl_seconds=60
    )
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(code)
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


async def test_a_code_naming_an_empty_user_is_refused(
    service: OAuthService,
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.aexchange_code(code_for(""))
    assert caught.value.error_enum is AuthErrors.OAUTH_CODE_INVALID


# Handing the code back to the browser


def test_the_success_redirect_gains_a_query_when_it_has_none(
    service: OAuthService,
) -> None:
    url = service.success_redirect_url("abc")
    assert url == "https://app.example.com/auth/callback?code=abc"


def test_the_success_redirect_keeps_a_query_it_already_had(
    make_identity_settings: Any,
) -> None:
    make_identity_settings(oauth_success_redirect="https://app.example.com/cb?next=/x")
    url = OAuthService().success_redirect_url("abc")
    assert url == "https://app.example.com/cb?next=/x&code=abc"


def test_the_code_is_escaped_into_the_redirect(service: OAuthService) -> None:
    """A code is base64url, but a redirect built by concatenation is a habit."""
    assert "a+b/c=" not in service.success_redirect_url("a+b/c=")


# The provider list the SPA renders


def test_enabled_providers_are_those_the_service_configured(
    service: OAuthService,
) -> None:
    assert service.enabled_providers() == ["github", "google"]


def test_a_provider_with_no_credentials_is_not_offered(
    make_identity_settings: Any,
) -> None:
    make_identity_settings(oauth_google_client_secret="")
    assert OAuthService().enabled_providers() == ["github"]


def test_a_service_offering_nothing_lists_nothing(
    make_identity_settings: Any,
) -> None:
    make_identity_settings(oauth_providers="")
    assert OAuthService().enabled_providers() == []
