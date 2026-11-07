"""PKCE and password hashing — the two primitives the auth routes lean on.

PKCE is what stops an intercepted authorization code being redeemed by anyone
but the client that started the flow, so the challenge must really be the
SHA-256 of the verifier and the verifier must really be unguessable. Password
verification must answer false rather than raise, whatever it is handed.
"""

import base64
import hashlib

import pytest

from base.tools.security import PasswordHashingService, b64url, pkce_pair

URL_SAFE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def test_b64url_drops_the_padding() -> None:
    assert b64url(b"a") == "YQ"
    assert b64url(b"ab") == "YWI"
    assert b64url(b"abc") == "YWJj"
    assert "=" not in b64url(b"a")


def test_b64url_uses_the_url_safe_alphabet() -> None:
    """`+` and `/` would have to be escaped in a query string."""
    raw = bytes([251, 255, 190])
    assert base64.b64encode(raw).decode() == "+/++"
    assert b64url(raw) == "-_--"


def test_b64url_of_nothing_is_an_empty_string() -> None:
    assert b64url(b"") == ""


def test_b64url_round_trips_through_the_standard_decoder() -> None:
    raw = bytes(range(32))
    encoded = b64url(raw)
    padded = encoded + "=" * (-len(encoded) % 4)
    assert base64.urlsafe_b64decode(padded) == raw


def test_a_pkce_challenge_is_the_sha256_of_its_verifier() -> None:
    verifier, challenge = pkce_pair()
    assert challenge == b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def test_a_pkce_verifier_carries_256_bits_of_entropy() -> None:
    verifier, _ = pkce_pair()
    assert len(verifier) == 43  # 32 bytes, base64url, unpadded


def test_a_pkce_challenge_is_a_43_character_s256_digest() -> None:
    _, challenge = pkce_pair()
    assert len(challenge) == 43


def test_a_pkce_pair_is_url_safe_throughout() -> None:
    """RFC 7636 allows only unreserved characters in either value."""
    verifier, challenge = pkce_pair()
    assert set(verifier) <= URL_SAFE
    assert set(challenge) <= URL_SAFE


def test_a_pkce_verifier_is_fresh_every_time() -> None:
    verifiers = {pkce_pair()[0] for _ in range(50)}
    assert len(verifiers) == 50


def test_a_pkce_challenge_does_not_reveal_its_verifier() -> None:
    verifier, challenge = pkce_pair()
    assert verifier != challenge
    assert verifier not in challenge


def test_a_password_hash_is_argon2id() -> None:
    hashed = PasswordHashingService.hash("correct horse battery staple")
    assert hashed.startswith("$argon2id$")


def test_a_password_verifies_against_its_own_hash() -> None:
    hashed = PasswordHashingService.hash("correct horse battery staple")
    assert PasswordHashingService.verify("correct horse battery staple", hashed) is True


def test_a_different_password_does_not_verify() -> None:
    hashed = PasswordHashingService.hash("correct horse battery staple")
    assert (
        PasswordHashingService.verify("Correct horse battery staple", hashed) is False
    )


def test_the_same_password_hashes_differently_each_time() -> None:
    """A per-hash salt, so two users sharing a password are not visibly equal."""
    first = PasswordHashingService.hash("same-password")
    second = PasswordHashingService.hash("same-password")
    assert first != second
    assert PasswordHashingService.verify("same-password", second) is True


@pytest.mark.parametrize(
    "stored",
    ["", "not-a-hash", "$argon2id$", "$2b$12$abcdefghijklmnopqrstuv"],
    ids=["empty", "plain", "truncated", "bcrypt"],
)
def test_an_unusable_stored_hash_answers_false_rather_than_raising(
    stored: str,
) -> None:
    """Every argon2 failure mode is one answer: the password did not match."""
    assert PasswordHashingService.verify("any-password", stored) is False


def test_an_empty_password_still_hashes_and_verifies() -> None:
    hashed = PasswordHashingService.hash("")
    assert PasswordHashingService.verify("", hashed) is True
    assert PasswordHashingService.verify("x", hashed) is False


def test_the_service_is_never_instantiated() -> None:
    """`NonInitialize` keeps one hasher rather than one per call site."""
    assert PasswordHashingService() is PasswordHashingService
