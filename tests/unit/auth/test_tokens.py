"""Issuing and verifying first-party tokens.

Pins the claims a token carries, the expiry arithmetic the two expiry settings
drive, and every way a presented token is refused — because a verifier that
accepts one of these is the whole of an authentication bypass.
"""

from datetime import UTC, datetime
from typing import Any

import jwt
import pytest

from base.models.definitions.errors import AuthErrors
from base.services.auth.models import AccessToken, Principal, RefreshToken, TokenPair
from base.services.auth.tokens import TokenIssuer, TokenVerifier
from base.services.configuration.mixins import (
    TokenIssuingSettingsMixin,
    TokenVerificationSettingsMixin,
)
from base.tools.errors import ErrorsException, InvalidSignedPayload
from tests.unit.auth.conftest import KeyPair

CLAIMS: dict[str, Any] = {
    "user_id": "11111111-1111-1111-1111-111111111111",
    "email": "someone@example.com",
    "roles": ["admin", "reader"],
    "scope": "read write",
}


def _issuing_settings(make_settings: Any, keys: KeyPair, **values: Any) -> Any:
    return make_settings(
        TokenIssuingSettingsMixin,
        auth_private_key_path=str(keys.private_path),
        auth_public_key_path=str(keys.public_path),
        **values,
    )


def test_an_access_token_carries_the_claims_it_was_given(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    token = TokenIssuer().issue_access(CLAIMS)
    payload = jwt.decode(token, keys.public_pem, algorithms=["RS256"])
    assert payload["user_id"] == CLAIMS["user_id"]
    assert payload["email"] == CLAIMS["email"]
    assert payload["roles"] == CLAIMS["roles"]
    assert payload["type"] == "access"


def test_a_refresh_token_is_marked_as_one(make_settings: Any, keys: KeyPair) -> None:
    _issuing_settings(make_settings, keys)
    token = TokenIssuer().issue_refresh(CLAIMS)
    payload = jwt.decode(token, keys.public_pem, algorithms=["RS256"])
    assert payload["type"] == "refresh"


def test_a_pair_drops_the_email_from_the_refresh_token(
    make_settings: Any, keys: KeyPair
) -> None:
    """A refresh token is stored for a week; it carries no address."""
    _issuing_settings(make_settings, keys)
    pair = TokenIssuer().issue_pair(CLAIMS)
    access = jwt.decode(pair.access_token, keys.public_pem, algorithms=["RS256"])
    refresh = jwt.decode(pair.refresh_token, keys.public_pem, algorithms=["RS256"])
    assert access["email"] == CLAIMS["email"]
    assert "email" not in refresh
    assert refresh["user_id"] == CLAIMS["user_id"]


def test_a_pair_is_a_token_pair_of_both_token_types(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    pair = TokenIssuer().issue_pair(CLAIMS)
    assert isinstance(pair, TokenPair)
    assert isinstance(pair.access_token, str)
    assert isinstance(pair.refresh_token, str)


def test_a_token_is_signed_rs256(make_settings: Any, keys: KeyPair) -> None:
    _issuing_settings(make_settings, keys)
    token = TokenIssuer().issue_access(CLAIMS)
    assert jwt.get_unverified_header(token)["alg"] == "RS256"


def test_expiry_is_issued_at_plus_the_access_expiry_setting(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys, access_token_expiry_seconds=120)
    before = datetime.now(UTC).timestamp()
    token = TokenIssuer().issue_access(CLAIMS)
    after = datetime.now(UTC).timestamp()

    payload = jwt.decode(token, keys.public_pem, algorithms=["RS256"])
    # `iat` is written as a whole-second JWT timestamp, so compare truncated.
    assert int(before) <= payload["iat"] <= int(after) + 1
    assert before + 120 <= payload["exp"] <= after + 121


def test_expiry_is_issued_at_plus_the_refresh_expiry_setting(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys, refresh_token_expiry_seconds=86_400)
    before = datetime.now(UTC).timestamp()
    token = TokenIssuer().issue_refresh(CLAIMS)

    payload = jwt.decode(token, keys.public_pem, algorithms=["RS256"])
    assert payload["exp"] - payload["iat"] == pytest.approx(86_400, abs=1)
    assert payload["exp"] >= before + 86_400


def test_the_two_expiries_are_independent(make_settings: Any, keys: KeyPair) -> None:
    _issuing_settings(
        make_settings,
        keys,
        access_token_expiry_seconds=60,
        refresh_token_expiry_seconds=3600,
    )
    pair = TokenIssuer().issue_pair(CLAIMS)
    access = jwt.decode(pair.access_token, keys.public_pem, algorithms=["RS256"])
    refresh = jwt.decode(pair.refresh_token, keys.public_pem, algorithms=["RS256"])
    assert access["exp"] - access["iat"] == pytest.approx(60, abs=1)
    assert refresh["exp"] - refresh["iat"] == pytest.approx(3600, abs=1)


def test_the_default_expiries_are_fifteen_minutes_and_seven_days(
    make_settings: Any, keys: KeyPair
) -> None:
    settings = _issuing_settings(make_settings, keys)
    assert settings.access_token_expiry_seconds == 900
    assert settings.refresh_token_expiry_seconds == 604_800


def test_an_arbitrary_payload_can_be_signed_with_its_own_ttl(
    make_settings: Any, keys: KeyPair
) -> None:
    """How an OAuth state survives the round trip to a provider."""
    _issuing_settings(make_settings, keys)
    signed = TokenIssuer().sign_payload({"nonce": "abc"}, ttl_seconds=30)
    payload = jwt.decode(signed, keys.public_pem, algorithms=["RS256"])
    assert payload["nonce"] == "abc"
    assert payload["exp"] - payload["iat"] == pytest.approx(30, abs=1)
    assert "type" not in payload


def test_verifying_a_valid_access_token_returns_the_caller_it_asserts(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    token = TokenIssuer().issue_access(CLAIMS)

    principal = TokenVerifier().verify(token)
    assert isinstance(principal, Principal)
    assert principal.subject == CLAIMS["user_id"]
    assert str(principal.user_id) == CLAIMS["user_id"]
    assert principal.email == CLAIMS["email"]
    assert principal.roles == frozenset({"admin", "reader"})
    assert principal.scopes == frozenset({"read", "write"})
    assert principal.token_type == "access"
    assert principal.claims["type"] == "access"


def test_a_subject_that_is_not_a_uuid_leaves_user_id_unset(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    token = TokenIssuer().issue_access({"sub": "an-external-subject"})

    principal = TokenVerifier().verify(token)
    assert principal.subject == "an-external-subject"
    assert principal.user_id is None


def test_a_refresh_token_is_refused_where_an_access_token_is_expected(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    refresh = TokenIssuer().issue_refresh(CLAIMS)

    with pytest.raises(ErrorsException) as caught:
        TokenVerifier().verify(refresh)
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


def test_a_refresh_token_verifies_when_that_is_what_is_expected(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    refresh = TokenIssuer().issue_refresh(CLAIMS)

    principal = TokenVerifier().verify(refresh, expected_type="refresh")
    assert principal.token_type == "refresh"


def test_a_token_with_no_subject_claim_is_refused(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    token = TokenIssuer().issue_access({"email": "nobody@example.com"})

    with pytest.raises(ErrorsException) as caught:
        TokenVerifier().verify(token)
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


def test_an_expired_token_is_refused(make_settings: Any, keys: KeyPair) -> None:
    _issuing_settings(make_settings, keys, access_token_expiry_seconds=-10)
    token = TokenIssuer().issue_access(CLAIMS)

    with pytest.raises(ErrorsException) as caught:
        TokenVerifier().verify(token)
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_EXPIRED


def test_a_token_signed_by_another_key_is_refused(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    forged = jwt.encode(
        {**CLAIMS, "type": "access", "exp": datetime.now(UTC).timestamp() + 600},
        other_keys.private_pem,
        algorithm="RS256",
    )

    with pytest.raises(ErrorsException) as caught:
        TokenVerifier().verify(forged)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_a_token_signed_with_an_unsupported_algorithm_is_refused(
    make_settings: Any, keys: KeyPair
) -> None:
    """A symmetric token offered where only RS256 is accepted is a forgery."""
    _issuing_settings(make_settings, keys)
    forged = jwt.encode(
        {**CLAIMS, "type": "access", "exp": datetime.now(UTC).timestamp() + 600},
        "a-shared-secret-long-enough-for-sha256",
        algorithm="HS256",
    )

    with pytest.raises(ErrorsException) as caught:
        TokenVerifier().verify(forged)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_an_unsigned_token_is_refused(make_settings: Any, keys: KeyPair) -> None:
    _issuing_settings(make_settings, keys)
    unsigned = jwt.encode({**CLAIMS, "type": "access"}, key="", algorithm="none")

    with pytest.raises(ErrorsException) as caught:
        TokenVerifier().verify(unsigned)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


@pytest.mark.parametrize(
    "token",
    ["", "not-a-token", "a.b.c", "Bearer eyJ...", "....", "e30.e30."],
    ids=["empty", "plain", "three-junk-segments", "scheme-included", "dots", "stubs"],
)
def test_a_malformed_token_is_refused(
    make_settings: Any, keys: KeyPair, token: str
) -> None:
    _issuing_settings(make_settings, keys)
    with pytest.raises(ErrorsException) as caught:
        TokenVerifier().verify(token)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_none_is_refused_rather_than_crashing(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    with pytest.raises(ErrorsException) as caught:
        TokenVerifier().verify(None)  # type: ignore[arg-type]
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_decode_local_returns_raw_claims_for_the_expected_type(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    token = TokenIssuer().issue_refresh(CLAIMS)

    claims = TokenVerifier().decode_local(token, expected_type="refresh")
    assert claims["user_id"] == CLAIMS["user_id"]
    assert claims["type"] == "refresh"


def test_decode_payload_reports_an_expiry_apart_from_a_tampering(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    """`InvalidSignedPayload` keeps state errors out of the caller error codes."""
    _issuing_settings(make_settings, keys)
    issuer, verifier = TokenIssuer(), TokenVerifier()

    expired = issuer.sign_payload({"nonce": "abc"}, ttl_seconds=-1)
    with pytest.raises(InvalidSignedPayload) as caught:
        verifier.decode_payload(expired)
    assert caught.value.expired is True

    forged = jwt.encode(
        {"nonce": "abc", "exp": datetime.now(UTC).timestamp() + 600},
        other_keys.private_pem,
        algorithm="RS256",
    )
    with pytest.raises(InvalidSignedPayload) as caught:
        verifier.decode_payload(forged)
    assert caught.value.expired is False


def test_decode_payload_returns_the_claims_of_a_valid_payload(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    signed = TokenIssuer().sign_payload({"nonce": "abc"}, ttl_seconds=60)
    assert TokenVerifier().decode_payload(signed)["nonce"] == "abc"


def test_a_verifier_needs_only_the_public_key(
    make_settings: Any, keys: KeyPair
) -> None:
    """A service that verifies but never issues composes the smaller mixin."""
    make_settings(
        TokenVerificationSettingsMixin,
        auth_public_key_path=str(keys.public_path),
    )
    token = jwt.encode(
        {**CLAIMS, "type": "access", "exp": datetime.now(UTC).timestamp() + 600},
        keys.private_pem,
        algorithm="RS256",
    )
    assert TokenVerifier().verify(token).email == CLAIMS["email"]


def test_an_issuer_reads_its_private_key_once_at_construction(
    make_settings: Any, keys: KeyPair
) -> None:
    _issuing_settings(make_settings, keys)
    issuer = TokenIssuer()
    keys.private_path.unlink()
    assert issuer.issue_access(CLAIMS)


def test_a_missing_key_file_fails_at_construction(
    make_settings: Any, keys: KeyPair, tmp_path: Any
) -> None:
    make_settings(
        TokenIssuingSettingsMixin,
        auth_private_key_path=str(tmp_path / "absent.pem"),
        auth_public_key_path=str(keys.public_path),
    )
    with pytest.raises(FileNotFoundError):
        TokenIssuer()


def test_the_access_and_refresh_new_types_are_distinct_names_for_str() -> None:
    """A `NewType` is erased at runtime; the distinction is for the reader."""
    assert AccessToken.__name__ == "AccessToken"
    assert RefreshToken.__name__ == "RefreshToken"
    assert AccessToken is not RefreshToken
    assert AccessToken.__supertype__ is str
    assert RefreshToken.__supertype__ is str
    assert isinstance(AccessToken("x"), str)
