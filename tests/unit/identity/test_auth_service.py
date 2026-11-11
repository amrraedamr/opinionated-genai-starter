"""Login, refresh and the user a verified token stands for.

Issuing is where a mistake is silent: an access token that outlives its
refresh, or a refresh token carrying claims it has no business carrying.
Resolving is where the two authentication paths meet — a token this service
signed, and one an enterprise IdP signed — and only one of them may create an
account on arrival.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from base.models.definitions.errors import AuthErrors, DatabaseErrors
from base.models.definitions.tenancy import DEFAULT_TENANT, DEFAULT_TENANT_CLAIM
from base.models.definitions.tokens import TokenType
from base.models.view_models.auth import Principal
from base.services.auth import TokenIssuer, TokenVerifier
from base.tools.errors import ErrorsException
from base.tools.misc import parse_uuid
from identity.models.orm.user import UserDTO
from identity.models.view_models.auth import RefreshToken
from identity.services.auth import AuthService
from identity.stores.user import UserStore
from identity.stores.user_identity import UserIdentityStore

ISSUER = "https://login.microsoftonline.com/tenant/v2.0"


@pytest.fixture
def service(identity_settings: Any) -> AuthService:
    return AuthService()


@pytest.fixture
def idp_service(make_identity_settings: Any) -> AuthService:
    """An identity service that also accepts an enterprise IdP's tokens."""
    make_identity_settings(
        idp_resource_server_enabled=True,
        idp_issuer=ISSUER,
        idp_audience="api://identity",
        idp_jwks_url="https://login.microsoftonline.com/tenant/discovery/keys",
        idp_provider_name="entra",
    )
    return AuthService()


async def a_user(email: str = "person@example.com", **values: Any) -> UserDTO:
    return await UserStore().aregister(UserDTO(email=email, **values))


def principal(**overrides: Any) -> Principal:
    subject = overrides.pop("subject", None) or str(uuid.uuid4())
    values: dict[str, Any] = {
        "subject": subject,
        "user_id": overrides.pop("user_id", parse_uuid(subject)),
        "token_type": TokenType.ACCESS,
        **overrides,
    }
    return Principal(**values)


# Issuing


async def test_an_issued_pair_names_the_user_in_both_halves(
    service: AuthService,
) -> None:
    user = await a_user()
    pair = service.generate_token_pair(user)
    verifier = TokenVerifier()
    access = verifier.decode_local(pair.access_token, expected_type=TokenType.ACCESS)
    refresh = verifier.decode_local(pair.refresh_token, expected_type=TokenType.REFRESH)
    assert access["user_id"] == refresh["user_id"] == str(user.id)


async def test_the_access_token_carries_the_address_and_the_refresh_does_not(
    service: AuthService,
) -> None:
    """A refresh token is held for a week; it should assert as little as possible."""
    user = await a_user()
    pair = service.generate_token_pair(user)
    verifier = TokenVerifier()
    access = verifier.decode_local(pair.access_token, expected_type=TokenType.ACCESS)
    refresh = verifier.decode_local(pair.refresh_token, expected_type=TokenType.REFRESH)
    assert access["email"] == "person@example.com"
    assert "email" not in refresh


async def test_the_access_token_names_the_users_tenant(service: AuthService) -> None:
    user = await a_user(tenant_id="acme")
    access = TokenVerifier().decode_local(
        service.generate_token_pair(user).access_token, expected_type=TokenType.ACCESS
    )
    assert access[DEFAULT_TENANT_CLAIM] == "acme"


async def test_a_user_registered_without_a_tenant_is_in_the_default_one(
    service: AuthService,
) -> None:
    user = await a_user()
    assert user.tenant_id == DEFAULT_TENANT


async def test_the_two_halves_declare_different_types(service: AuthService) -> None:
    user = await a_user()
    pair = service.generate_token_pair(user)
    with pytest.raises(ErrorsException) as caught:
        TokenVerifier().decode_local(pair.access_token, expected_type=TokenType.REFRESH)
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


async def test_an_issued_pair_expires(service: AuthService) -> None:
    user = await a_user()
    pair = service.generate_token_pair(user)
    claims = TokenVerifier().decode_local(
        pair.access_token, expected_type=TokenType.ACCESS
    )
    assert claims["exp"] > claims["iat"]


# Refreshing


async def test_a_refresh_token_buys_a_new_pair(service: AuthService) -> None:
    user = await a_user()
    pair = service.generate_token_pair(user)
    renewed = await service.arefresh_access_token(RefreshToken(pair.refresh_token))
    claims = TokenVerifier().decode_local(
        renewed.access_token, expected_type=TokenType.ACCESS
    )
    assert claims["user_id"] == str(user.id)
    assert claims["email"] == "person@example.com"


async def test_an_access_token_cannot_be_refreshed_with(
    service: AuthService,
) -> None:
    user = await a_user()
    pair = service.generate_token_pair(user)
    with pytest.raises(ErrorsException) as caught:
        await service.arefresh_access_token(RefreshToken(pair.access_token))
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


async def test_a_refresh_token_this_service_never_signed_is_refused(
    service: AuthService,
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.arefresh_access_token(RefreshToken("nonsense"))
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


async def test_an_expired_refresh_token_says_so(service: AuthService) -> None:
    user = await a_user()
    expired = TokenIssuer().sign_payload(
        {"user_id": str(user.id), "type": TokenType.REFRESH}, ttl_seconds=-10
    )
    with pytest.raises(ErrorsException) as caught:
        await service.arefresh_access_token(RefreshToken(expired))
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_EXPIRED


async def test_a_refresh_token_for_a_deleted_user_is_a_record_not_found(
    service: AuthService,
) -> None:
    user = await a_user()
    pair = service.generate_token_pair(user)
    await UserStore().adelete(user.id)
    with pytest.raises(ErrorsException) as caught:
        await service.arefresh_access_token(RefreshToken(pair.refresh_token))
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND


async def test_a_refresh_token_naming_nobody_is_refused(
    service: AuthService,
) -> None:
    """Not reachable through issuing, which is no reason to crash on it."""
    forged = TokenIssuer().sign_payload({"type": TokenType.REFRESH}, ttl_seconds=60)
    with pytest.raises(ErrorsException) as caught:
        await service.arefresh_access_token(RefreshToken(forged))
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


async def test_a_refresh_token_naming_something_that_is_not_a_user_is_refused(
    service: AuthService,
) -> None:
    forged = TokenIssuer().sign_payload(
        {"user_id": "' OR 1=1 --", "type": TokenType.REFRESH}, ttl_seconds=60
    )
    with pytest.raises(ErrorsException) as caught:
        await service.arefresh_access_token(RefreshToken(forged))
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


# Resolving a first-party principal


async def test_a_principal_resolves_to_the_user_it_names(
    service: AuthService,
) -> None:
    user = await a_user()
    resolved = await service.aresolve_principal(
        principal(subject=str(user.id), user_id=user.id)
    )
    assert resolved.id == user.id


async def test_a_principal_naming_nobody_is_refused(service: AuthService) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.aresolve_principal(principal(subject="not-a-uuid", user_id=None))
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


async def test_a_principal_naming_a_user_who_is_gone_is_a_record_not_found(
    service: AuthService,
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.aresolve_principal(principal())
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND


async def test_an_issuer_is_ignored_while_the_idp_is_switched_off(
    service: AuthService,
) -> None:
    """Claiming to be the IdP buys nothing from a service that accepts none."""
    user = await a_user()
    resolved = await service.aresolve_principal(
        principal(subject=str(user.id), user_id=user.id, issuer=ISSUER)
    )
    assert resolved.id == user.id


# Resolving an IdP principal


async def test_an_idp_principal_is_linked_on_arrival(
    idp_service: AuthService,
) -> None:
    resolved = await idp_service.aresolve_principal(
        principal(
            subject="entra-subject",
            user_id=None,
            email="person@example.com",
            issuer=ISSUER,
            claims={"given_name": "Ada", "family_name": "Lovelace"},
        )
    )
    assert resolved.email == "person@example.com"
    assert resolved.first_name == "Ada"
    link = await UserIdentityStore().aget_by_provider_subject("entra", "entra-subject")
    assert link is not None


async def test_an_idp_principal_with_no_address_is_refused(
    idp_service: AuthService,
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await idp_service.aresolve_principal(
            principal(subject="entra-subject", user_id=None, issuer=ISSUER)
        )
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


async def test_an_idp_principal_reaching_an_existing_account_links_to_it(
    idp_service: AuthService,
) -> None:
    existing = await a_user(password_hash="argon2")
    resolved = await idp_service.aresolve_principal(
        principal(
            subject="entra-subject",
            user_id=None,
            email="person@example.com",
            issuer=ISSUER,
        )
    )
    assert resolved.id == existing.id


async def test_an_idp_that_marks_an_address_unverified_cannot_take_the_account(
    idp_service: AuthService,
) -> None:
    await a_user(password_hash="argon2")
    with pytest.raises(ErrorsException) as caught:
        await idp_service.aresolve_principal(
            principal(
                subject="entra-subject",
                user_id=None,
                email="person@example.com",
                issuer=ISSUER,
                claims={"email_verified": False},
            )
        )
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


async def test_a_token_from_another_issuer_takes_the_first_party_path(
    idp_service: AuthService,
) -> None:
    """Only the configured issuer gets to create an account by arriving."""
    with pytest.raises(ErrorsException) as caught:
        await idp_service.aresolve_principal(
            principal(
                subject="someone",
                user_id=None,
                email="person@example.com",
                issuer="https://evil.example.com",
            )
        )
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


async def test_a_token_with_no_issuer_takes_the_first_party_path(
    idp_service: AuthService,
) -> None:
    user = await a_user()
    resolved = await idp_service.aresolve_principal(
        principal(subject=str(user.id), user_id=user.id)
    )
    assert resolved.id == user.id
