"""Apple's client secret, which is a signed assertion rather than a stored string.

Sign in with Apple issues no static secret: one is minted per exchange from
the team id, key id and private key, signed ES256 and short-lived. Pins the
claims Apple checks and the ways the signing material can be supplied.
"""

import time
from typing import Any

import jwt
import pytest

from base.services.auth.oauth.apple_secret import (
    build_apple_client_secret,
    load_apple_private_key,
    resolve_apple_client_secret,
)
from base.services.configuration.mixins import AppleOAuthSettingsMixin
from tests.conftest import settings_class

APPLE_AUDIENCE = "https://appleid.apple.com"


def _apple_settings(**values: Any) -> Any:
    return settings_class(AppleOAuthSettingsMixin)(service_name="svc", **values)


def test_the_secret_is_a_jwt_signed_es256_with_the_key_id_in_its_header(
    ec_private_pem: str,
) -> None:
    secret = build_apple_client_secret(
        client_id="com.example.service",
        team_id="TEAMID",
        key_id="KEYID",
        private_key_pem=ec_private_pem,
    )
    header = jwt.get_unverified_header(secret)
    assert header["alg"] == "ES256"
    assert header["kid"] == "KEYID"


def test_the_secret_claims_are_the_ones_apple_checks(
    ec_private_pem: str, ec_public_pem: str
) -> None:
    before = int(time.time())
    secret = build_apple_client_secret(
        client_id="com.example.service",
        team_id="TEAMID",
        key_id="KEYID",
        private_key_pem=ec_private_pem,
    )
    claims = jwt.decode(
        secret, ec_public_pem, algorithms=["ES256"], audience=APPLE_AUDIENCE
    )
    assert claims["iss"] == "TEAMID"
    assert claims["sub"] == "com.example.service"
    assert claims["aud"] == APPLE_AUDIENCE
    assert before <= claims["iat"] <= int(time.time())
    assert claims["exp"] - claims["iat"] == 86_400


def test_the_lifetime_is_caller_supplied(
    ec_private_pem: str, ec_public_pem: str
) -> None:
    secret = build_apple_client_secret(
        client_id="com.example.service",
        team_id="TEAMID",
        key_id="KEYID",
        private_key_pem=ec_private_pem,
        ttl_seconds=600,
    )
    claims = jwt.decode(
        secret, ec_public_pem, algorithms=["ES256"], audience=APPLE_AUDIENCE
    )
    assert claims["exp"] - claims["iat"] == 600


def test_the_secret_verifies_only_against_the_matching_public_key(
    ec_private_pem: str, ec_public_pem: str
) -> None:
    secret = build_apple_client_secret(
        client_id="com.example.service",
        team_id="TEAMID",
        key_id="KEYID",
        private_key_pem=ec_private_pem,
    )
    assert jwt.decode(
        secret, ec_public_pem, algorithms=["ES256"], audience=APPLE_AUDIENCE
    )
    with pytest.raises(jwt.InvalidAudienceError):
        jwt.decode(
            secret, ec_public_pem, algorithms=["ES256"], audience="https://example.com"
        )


def test_an_rsa_key_cannot_sign_an_es256_assertion(keys: Any) -> None:
    """Apple issues an EC key; handing it an RSA one must fail loudly."""
    with pytest.raises(Exception) as caught:
        build_apple_client_secret(
            client_id="com.example.service",
            team_id="TEAMID",
            key_id="KEYID",
            private_key_pem=keys.private_pem,
        )
    assert "EC" in str(caught.value) or "ECAlgorithm" in type(caught.value).__name__


def test_a_key_that_is_not_a_key_fails_clearly(ec_private_pem: str) -> None:
    with pytest.raises(Exception) as caught:
        build_apple_client_secret(
            client_id="com.example.service",
            team_id="TEAMID",
            key_id="KEYID",
            private_key_pem="-----BEGIN PRIVATE KEY-----\nnot-a-key\n",
        )
    assert str(caught.value)


def test_an_empty_key_fails_rather_than_signing_nothing() -> None:
    with pytest.raises(Exception):
        build_apple_client_secret(
            client_id="com.example.service",
            team_id="TEAMID",
            key_id="KEYID",
            private_key_pem="",
        )


def test_an_inline_private_key_is_read_and_stripped(ec_private_pem: str) -> None:
    config = _apple_settings(oauth_apple_private_key=ec_private_pem)
    assert load_apple_private_key(config) == ec_private_pem.strip()


def test_an_inline_private_key_with_escaped_newlines_is_restored(
    ec_private_pem: str,
) -> None:
    """How a PEM survives a single-line environment variable."""
    one_line = ec_private_pem.strip().replace("\n", "\\n")
    config = _apple_settings(oauth_apple_private_key=one_line)
    assert load_apple_private_key(config) == ec_private_pem.strip()


def test_a_private_key_path_is_read_when_no_inline_key_is_set(
    ec_private_pem: str, tmp_path: Any
) -> None:
    path = tmp_path / "apple.p8"
    path.write_text(ec_private_pem)
    config = _apple_settings(oauth_apple_private_key_path=str(path))
    assert load_apple_private_key(config) == ec_private_pem


def test_an_inline_key_wins_over_a_path(ec_private_pem: str, tmp_path: Any) -> None:
    path = tmp_path / "apple.p8"
    path.write_text("from-the-file")
    config = _apple_settings(
        oauth_apple_private_key=ec_private_pem,
        oauth_apple_private_key_path=str(path),
    )
    assert load_apple_private_key(config) == ec_private_pem.strip()


def test_no_signing_material_loads_as_an_empty_key() -> None:
    assert load_apple_private_key(_apple_settings()) == ""


def test_a_missing_key_file_is_reported(tmp_path: Any) -> None:
    config = _apple_settings(oauth_apple_private_key_path=str(tmp_path / "absent.p8"))
    with pytest.raises(FileNotFoundError):
        load_apple_private_key(config)


def test_a_configured_client_secret_is_used_as_is(ec_private_pem: str) -> None:
    config = _apple_settings(
        oauth_apple_client_id="com.example.service",
        oauth_apple_client_secret="pasted-in-by-hand",
        oauth_apple_team_id="TEAMID",
        oauth_apple_key_id="KEYID",
        oauth_apple_private_key=ec_private_pem,
    )
    assert resolve_apple_client_secret(config) == "pasted-in-by-hand"


def test_a_secret_is_minted_from_complete_signing_material(
    ec_private_pem: str, ec_public_pem: str
) -> None:
    config = _apple_settings(
        oauth_apple_client_id="com.example.service",
        oauth_apple_team_id="TEAMID",
        oauth_apple_key_id="KEYID",
        oauth_apple_private_key=ec_private_pem,
    )
    claims = jwt.decode(
        resolve_apple_client_secret(config),
        ec_public_pem,
        algorithms=["ES256"],
        audience=APPLE_AUDIENCE,
    )
    assert claims["sub"] == "com.example.service"
    assert claims["iss"] == "TEAMID"


@pytest.mark.parametrize(
    "missing",
    [
        "oauth_apple_client_id",
        "oauth_apple_team_id",
        "oauth_apple_key_id",
        "oauth_apple_private_key",
    ],
)
def test_incomplete_signing_material_resolves_to_no_secret(
    missing: str, ec_private_pem: str
) -> None:
    values = {
        "oauth_apple_client_id": "com.example.service",
        "oauth_apple_team_id": "TEAMID",
        "oauth_apple_key_id": "KEYID",
        "oauth_apple_private_key": ec_private_pem,
    }
    values[missing] = ""
    assert resolve_apple_client_secret(_apple_settings(**values)) == ""


def test_each_minted_secret_is_fresh(
    ec_private_pem: str, ec_public_pem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minted per exchange, so an expiry never outlives the settings object."""
    config = _apple_settings(
        oauth_apple_client_id="com.example.service",
        oauth_apple_team_id="TEAMID",
        oauth_apple_key_id="KEYID",
        oauth_apple_private_key=ec_private_pem,
    )
    first = resolve_apple_client_secret(config)
    monkeypatch.setattr(time, "time", lambda: time.monotonic() + 10_000)
    second = resolve_apple_client_secret(config)

    assert first != second
    first_claims = jwt.decode(
        first, ec_public_pem, algorithms=["ES256"], audience=APPLE_AUDIENCE
    )
    second_claims = jwt.decode(
        second,
        ec_public_pem,
        algorithms=["ES256"],
        audience=APPLE_AUDIENCE,
        options={"verify_exp": False},
    )
    assert second_claims["iat"] != first_claims["iat"]
