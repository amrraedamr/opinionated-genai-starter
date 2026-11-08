"""Accepting an enterprise IdP's access tokens alongside first-party ones.

A resource server verifies tokens it did not mint, against keys it fetches.
Pins how the JWKS endpoint is found, which algorithms are allowed, and that a
token the IdP did not issue falls through to local verification rather than
being waved past.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from jwt.exceptions import ExpiredSignatureError, PyJWKClientError

import base.services.http as http_module
from base.models.definitions.errors import AuthErrors
from base.models.view_models.http import HttpResponse
from base.services.auth.tokens import TokenIssuer, TokenVerifier
from base.services.configuration.mixins import (
    IdpSettingsMixin,
    TokenIssuingSettingsMixin,
)
from base.tools.errors import ErrorsException
from tests.unit.auth.conftest import KeyPair

ISSUER = "https://login.example.com/tenant/v2.0"
AUDIENCE = "api://this-service"


class FakeJWKSClient:
    """Answers with one signing key, and records what it was asked about."""

    def __init__(self, key: str | None = None, error: Exception | None = None) -> None:
        self.key = key
        self.error = error
        self.tokens: list[str] = []

    def get_signing_key_from_jwt(self, token: str) -> Any:
        self.tokens.append(token)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(key=self.key)


def _idp_settings(make_settings: Any, keys: KeyPair, **values: Any) -> Any:
    values.setdefault("idp_resource_server_enabled", True)
    values.setdefault("idp_issuer", ISSUER)
    values.setdefault("idp_audience", AUDIENCE)
    values.setdefault("idp_jwks_url", "https://login.example.com/keys")
    return make_settings(
        TokenIssuingSettingsMixin,
        IdpSettingsMixin,
        auth_private_key_path=str(keys.private_path),
        auth_public_key_path=str(keys.public_path),
        **values,
    )


def _idp_token(
    private_pem: str,
    *,
    algorithm: str = "RS256",
    issuer: str = ISSUER,
    audience: str = AUDIENCE,
    expires_in: int = 600,
    **claims: Any,
) -> str:
    payload: dict[str, Any] = {
        "sub": "idp-subject",
        "email": "person@example.com",
        "iss": issuer,
        "aud": audience,
        "exp": datetime.now(UTC) + timedelta(seconds=expires_in),
        "iat": datetime.now(UTC),
    }
    payload.update(claims)
    return jwt.encode(payload, private_pem, algorithm=algorithm)


def _verifier_with(keys: KeyPair, client: FakeJWKSClient) -> TokenVerifier:
    verifier = TokenVerifier()
    verifier._jwks_client = client  # type: ignore[assignment]
    return verifier


def test_enabling_the_resource_server_without_an_issuer_refuses_to_start(
    make_settings: Any, keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys, idp_issuer="")
    with pytest.raises(ValueError, match="IDP_ISSUER and IDP_AUDIENCE"):
        TokenVerifier()


def test_enabling_the_resource_server_without_an_audience_refuses_to_start(
    make_settings: Any, keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys, idp_audience="")
    with pytest.raises(ValueError, match="IDP_ISSUER and IDP_AUDIENCE"):
        TokenVerifier()


def test_an_explicit_jwks_url_is_used_without_a_discovery_request(
    make_settings: Any, keys: KeyPair, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idp_settings(make_settings, keys, idp_jwks_url="https://keys.example.com/jwks")
    monkeypatch.setattr(
        http_module.HttpRequestService,
        "call_endpoint",
        _refuse_http,
        raising=True,
    )
    verifier = TokenVerifier()
    assert verifier._jwks_client is not None
    assert verifier._jwks_client.uri == "https://keys.example.com/jwks"


def test_a_discovery_document_supplies_the_jwks_url(
    make_settings: Any, keys: KeyPair, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idp_settings(
        make_settings,
        keys,
        idp_jwks_url="",
        idp_discovery_url="https://login.example.com/.well-known/openid-configuration",
    )
    asked = _stub_discovery(
        monkeypatch, HttpResponse.ok(200, {"jwks_uri": "https://discovered/jwks"})
    )

    verifier = TokenVerifier()
    assert verifier._jwks_client is not None
    assert verifier._jwks_client.uri == "https://discovered/jwks"
    assert asked == ["https://login.example.com/.well-known/openid-configuration"]


def test_the_issuer_is_asked_for_its_discovery_document_as_a_last_resort(
    make_settings: Any, keys: KeyPair, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idp_settings(make_settings, keys, idp_jwks_url="", idp_discovery_url="")
    asked = _stub_discovery(
        monkeypatch, HttpResponse.ok(200, {"jwks_uri": "https://from-issuer/jwks"})
    )

    TokenVerifier()
    assert asked == [f"{ISSUER}/.well-known/openid-configuration"]


def test_a_failed_discovery_request_refuses_to_start(
    make_settings: Any, keys: KeyPair, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idp_settings(
        make_settings,
        keys,
        idp_jwks_url="",
        idp_discovery_url="https://login.example.com/.well-known/openid-configuration",
    )
    _stub_discovery(monkeypatch, HttpResponse.fail(RuntimeError("down"), 503))

    with pytest.raises(ValueError, match="Failed to load IdP discovery document"):
        TokenVerifier()


def test_a_discovery_document_without_a_jwks_uri_refuses_to_start(
    make_settings: Any, keys: KeyPair, monkeypatch: pytest.MonkeyPatch
) -> None:
    _idp_settings(
        make_settings,
        keys,
        idp_jwks_url="",
        idp_discovery_url="https://login.example.com/.well-known/openid-configuration",
    )
    _stub_discovery(monkeypatch, HttpResponse.ok(200, {"issuer": ISSUER}))

    with pytest.raises(ValueError, match="missing jwks_uri"):
        TokenVerifier()


def test_settings_with_no_idp_mixin_leave_the_resource_server_off(
    make_settings: Any, keys: KeyPair
) -> None:
    make_settings(
        TokenIssuingSettingsMixin,
        auth_private_key_path=str(keys.private_path),
        auth_public_key_path=str(keys.public_path),
    )
    verifier = TokenVerifier()
    assert verifier._idp is None
    assert verifier._idp_enabled is False
    assert verifier._jwks_client is None


def test_an_idp_token_is_verified_against_the_jwks_key(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys)
    client = FakeJWKSClient(key=other_keys.public_pem)
    token = _idp_token(other_keys.private_pem, roles=["ops"], scope="read")

    principal = _verifier_with(keys, client).verify(token)
    assert principal.subject == "idp-subject"
    assert principal.email == "person@example.com"
    assert principal.issuer == ISSUER
    assert principal.token_type == "access"
    assert principal.roles == frozenset({"ops"})
    assert principal.scopes == frozenset({"read"})
    assert client.tokens == [token]


def test_a_first_party_token_still_verifies_while_the_idp_is_enabled(
    make_settings: Any, keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys)
    local = TokenIssuer().issue_access({"user_id": "local-subject"})

    principal = _verifier_with(keys, FakeJWKSClient()).verify(local)
    assert principal.subject == "local-subject"
    assert principal.token_type == "access"


def test_a_token_from_another_issuer_is_not_taken_for_an_idp_token(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    """It falls through to local verification, which refuses it."""
    _idp_settings(make_settings, keys)
    foreign = _idp_token(other_keys.private_pem, issuer="https://somewhere-else")
    client = FakeJWKSClient(key=other_keys.public_pem)

    with pytest.raises(ErrorsException) as caught:
        _verifier_with(keys, client).verify(foreign)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED
    assert client.tokens == []


def test_a_malformed_token_is_not_taken_for_an_idp_token(
    make_settings: Any, keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient())
    assert verifier._is_idp_token("not-a-token") is False


def test_a_token_from_another_issuer_is_refused(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    """A rejected claim is this library's error, not PyJWT's.

    Issuer, audience and missing-claim failures descend from
    `InvalidTokenError` rather than `DecodeError`, so naming `DecodeError`
    let them reach the caller raw — a 500 where a 401 was meant.
    """
    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    foreign = _idp_token(other_keys.private_pem, issuer="https://somewhere-else")

    with pytest.raises(ErrorsException) as caught:
        verifier._verify_idp(foreign)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_a_token_minted_for_another_audience_is_refused(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    """The one that matters most: a valid token for a different service."""
    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    token = _idp_token(other_keys.private_pem, audience="api://some-other-service")

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(token)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_a_token_with_no_audience_claim_is_refused(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    """A claim the verifier required and the token never carried."""
    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    token = jwt.encode(
        {
            "sub": "idp-subject",
            "email": "person@example.com",
            "iss": ISSUER,
            "exp": datetime.now(UTC) + timedelta(seconds=600),
        },
        other_keys.private_pem,
        algorithm="RS256",
    )

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(token)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_an_expired_idp_token_is_refused_as_expired(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    token = _idp_token(other_keys.private_pem, expires_in=-30)

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(token)
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_EXPIRED


def test_an_idp_token_signed_by_an_unknown_key_is_refused(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=keys.public_pem))
    token = _idp_token(other_keys.private_pem)

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(token)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_a_jwks_lookup_failure_is_refused_rather_than_raised_raw(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys)
    client = FakeJWKSClient(error=PyJWKClientError("no matching kid"))
    verifier = _verifier_with(keys, client)

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(_idp_token(other_keys.private_pem))
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_an_idp_token_without_a_subject_is_refused(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(_idp_token(other_keys.private_pem, sub=""))
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


def test_an_idp_token_without_the_configured_email_claim_is_refused(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    token = _idp_token(other_keys.private_pem, email=None)

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(token)
    assert caught.value.error_enum is AuthErrors.USER_TOKEN_INCOMPLETE


def test_the_email_claim_name_is_configurable(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys, idp_email_claim="preferred_username")
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    token = _idp_token(
        other_keys.private_pem, email=None, preferred_username="upn@example.com"
    )

    assert verifier.verify(token).email == "upn@example.com"


def test_the_allowed_algorithms_come_from_the_setting(
    make_settings: Any, keys: KeyPair, ec_private_pem: str, ec_public_pem: str
) -> None:
    """Spacing around a name is dropped; the name itself is passed through."""
    _idp_settings(make_settings, keys, idp_algorithms="  ES256  ")
    verifier = _verifier_with(keys, FakeJWKSClient(key=ec_public_pem))
    token = _idp_token(ec_private_pem, algorithm="ES256")

    assert verifier.verify(token).subject == "idp-subject"


def test_several_algorithms_can_be_allowed_at_once(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair, ec_private_pem: str
) -> None:
    _idp_settings(make_settings, keys, idp_algorithms="RS256,ES256")
    rsa_client = FakeJWKSClient(key=other_keys.public_pem)
    assert (
        _verifier_with(keys, rsa_client).verify(_idp_token(other_keys.private_pem))
        is not None
    )


def test_an_algorithm_name_in_the_wrong_case_is_refused(
    make_settings: Any, keys: KeyPair, ec_private_pem: str, ec_public_pem: str
) -> None:
    """`split_csv` is called without `lower`, so names are not normalised."""
    _idp_settings(make_settings, keys, idp_algorithms="es256")
    verifier = _verifier_with(keys, FakeJWKSClient(key=ec_public_pem))

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(_idp_token(ec_private_pem, algorithm="ES256"))
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_an_algorithm_outside_the_setting_is_refused(
    make_settings: Any, keys: KeyPair, ec_private_pem: str, ec_public_pem: str
) -> None:
    _idp_settings(make_settings, keys, idp_algorithms="RS256")
    verifier = _verifier_with(keys, FakeJWKSClient(key=ec_public_pem))

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(_idp_token(ec_private_pem, algorithm="ES256"))
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_an_empty_algorithms_setting_falls_back_to_rs256(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys, idp_algorithms="")
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))

    assert verifier.verify(_idp_token(other_keys.private_pem)).subject == "idp-subject"


def test_the_algorithms_setting_defaults_to_rs256(
    make_settings: Any, keys: KeyPair
) -> None:
    settings = _idp_settings(make_settings, keys)
    assert settings.idp_algorithms == "RS256"


def test_a_verifier_with_no_jwks_client_says_the_idp_is_not_configured(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    _idp_settings(make_settings, keys)
    verifier = TokenVerifier()
    verifier._jwks_client = None

    with pytest.raises(ErrorsException) as caught:
        verifier._verify_idp(_idp_token(other_keys.private_pem))
    assert caught.value.error_enum is AuthErrors.IDP_NOT_CONFIGURED


def _refuse_http(*args: Any, **kwargs: Any) -> HttpResponse:
    raise AssertionError("no HTTP request should have been made")


def _stub_discovery(
    monkeypatch: pytest.MonkeyPatch, response: HttpResponse
) -> list[str]:
    """Answer the discovery lookup, recording the URL it asked for."""
    asked: list[str] = []

    def _call(self: Any, url: str, *args: Any, **kwargs: Any) -> HttpResponse:
        asked.append(url)
        return response

    monkeypatch.setattr(http_module.HttpRequestService, "call_endpoint", _call)
    return asked


def test_a_token_not_yet_valid_is_refused(
    make_settings: Any, keys: KeyPair, other_keys: KeyPair
) -> None:
    """`nbf` in the future. Its error is a sibling of the ones above.

    Naming exception types one at a time left this whole family uncaught;
    catching `InvalidTokenError` covers the ones nobody thought to list.
    """
    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    token = jwt.encode(
        {
            "sub": "idp-subject",
            "email": "person@example.com",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "nbf": datetime.now(UTC) + timedelta(seconds=600),
            "exp": datetime.now(UTC) + timedelta(seconds=1200),
        },
        other_keys.private_pem,
        algorithm="RS256",
    )

    with pytest.raises(ErrorsException) as caught:
        verifier.verify(token)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_every_claim_failure_pyjwt_can_raise_is_mapped(
    make_settings: Any,
    keys: KeyPair,
    other_keys: KeyPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No member of the family may reach a caller as a raw PyJWT error.

    Walking the tree rather than naming members is what stops the next
    version of PyJWT from reopening the gap this closed: a claim check added
    there arrives here already mapped.
    """
    from jwt.exceptions import InvalidTokenError

    import base.services.auth.tokens as tokens_module

    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    token = _idp_token(other_keys.private_pem)

    family: list[type[BaseException]] = []

    def walk(cls: type[BaseException]) -> None:
        for sub in cls.__subclasses__():
            family.append(sub)
            walk(sub)

    walk(InvalidTokenError)
    assert len(family) > 8, "the family should not be this small"

    for cls in family:
        if issubclass(cls, ExpiredSignatureError):
            continue
        try:
            failure = cls("simulated")
        except Exception:
            continue

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise failure

        monkeypatch.setattr(tokens_module, "decode", refuse)
        with pytest.raises(ErrorsException) as caught:
            verifier._verify_idp(token)
        assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED, cls.__name__


def test_a_malformed_jwks_document_is_refused(
    make_settings: Any,
    keys: KeyPair,
    other_keys: KeyPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`PyJWKSetError` sits under neither family the handler named.

    It descends straight from `PyJWTError`, so an identity provider serving a
    broken key set produced a 500 rather than a refusal.
    """
    from jwt.exceptions import PyJWKSetError

    import base.services.auth.tokens as tokens_module

    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    token = _idp_token(other_keys.private_pem)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise PyJWKSetError("the key set is not valid")

    monkeypatch.setattr(tokens_module, "decode", refuse)
    with pytest.raises(ErrorsException) as caught:
        verifier._verify_idp(token)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED


def test_a_key_the_algorithm_cannot_use_is_refused(
    make_settings: Any,
    keys: KeyPair,
    other_keys: KeyPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forged HS256 token makes PyJWT raise `TypeError`, not a JWT error.

    The JWKS client hands back a key object; `HMACAlgorithm.prepare_key`
    wants a string and says so in a `TypeError`. Refusing the token is right,
    but it has to be refused rather than escape.
    """
    import base.services.auth.tokens as tokens_module

    _idp_settings(make_settings, keys)
    verifier = _verifier_with(keys, FakeJWKSClient(key=other_keys.public_pem))
    token = _idp_token(other_keys.private_pem)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("Expected a string value")

    monkeypatch.setattr(tokens_module, "decode", refuse)
    with pytest.raises(ErrorsException) as caught:
        verifier._verify_idp(token)
    assert caught.value.error_enum is AuthErrors.USER_DECODE_FAILED
