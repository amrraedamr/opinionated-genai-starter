"""`tools/security.py`: PKCE pairs and password hashing.

Both are small, and both fail quietly if they are wrong: a challenge that does
not derive from its verifier, or a verify that returns True on a bad hash.
"""

import base64
import hashlib

from base.tools.security import PasswordHashingService, b64url, pkce_pair


def test_base64url_drops_the_padding_a_url_cannot_carry() -> None:
    assert b64url(b"hello") == "aGVsbG8"
    assert "=" not in b64url(b"a")


def test_base64url_uses_the_url_safe_alphabet() -> None:
    encoded = b64url(bytes(range(256)))
    assert "+" not in encoded
    assert "/" not in encoded


def test_a_pkce_challenge_is_the_sha256_of_its_verifier() -> None:
    verifier, challenge = pkce_pair()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=")
    assert challenge == expected.decode("ascii")


def test_every_pkce_pair_is_new() -> None:
    first, _ = pkce_pair()
    second, _ = pkce_pair()
    assert first != second


def test_a_pkce_verifier_is_long_enough_for_the_spec() -> None:
    verifier, _ = pkce_pair()
    assert 43 <= len(verifier) <= 128


def test_a_password_verifies_against_its_own_hash() -> None:
    hashed = PasswordHashingService.hash("correct horse battery staple")
    assert PasswordHashingService.verify("correct horse battery staple", hashed) is True


def test_a_wrong_password_does_not_verify() -> None:
    hashed = PasswordHashingService.hash("right")
    assert PasswordHashingService.verify("wrong", hashed) is False


def test_hashing_the_same_password_twice_gives_different_hashes() -> None:
    first = PasswordHashingService.hash("same")
    second = PasswordHashingService.hash("same")
    assert first != second
    assert PasswordHashingService.verify("same", second) is True


def test_a_hash_that_is_not_a_hash_is_refused_rather_than_raising() -> None:
    assert PasswordHashingService.verify("any", "not-an-argon2-hash") is False
    assert PasswordHashingService.verify("any", "") is False


def test_the_hash_names_the_algorithm_it_used() -> None:
    assert PasswordHashingService.hash("x").startswith("$argon2")


def test_an_empty_password_still_round_trips() -> None:
    hashed = PasswordHashingService.hash("")
    assert PasswordHashingService.verify("", hashed) is True
    assert PasswordHashingService.verify("x", hashed) is False
