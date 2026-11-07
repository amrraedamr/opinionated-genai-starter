"""Keys and isolation the auth tests share.

`TokenIssuer` and `TokenVerifier` carry the `Singleton` metaclass, so one
built in an earlier test would answer in a later one with the earlier test's
settings. Every test here starts from an empty singleton registry.

Keys are generated in-process rather than committed: nothing in this tree is
ever a real credential, and an RSA pair is cheap enough once per session.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from base.services.common.meta import Singleton


@dataclass(frozen=True, slots=True)
class KeyPair:
    """A PEM key pair, both as text and as the files settings point at."""

    private_pem: str
    public_pem: str
    private_path: Path
    public_path: Path


def _generate_rsa() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_pem, public_pem


@pytest.fixture(autouse=True)
def _isolated_singletons() -> Iterator[None]:
    """Drop every cached singleton around each test."""
    Singleton.purge()
    yield
    Singleton.purge()


@pytest.fixture(scope="session")
def rsa_pem() -> tuple[str, str]:
    """One RSA pair for the whole session; generating it is the slow part."""
    return _generate_rsa()


@pytest.fixture(scope="session")
def other_rsa_pem() -> tuple[str, str]:
    """A second pair, for proving a wrong signature is rejected."""
    return _generate_rsa()


@pytest.fixture
def keys(rsa_pem: tuple[str, str], tmp_path: Path) -> KeyPair:
    """The session's RSA pair, written where settings can point at it."""
    private_pem, public_pem = rsa_pem
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_text(private_pem)
    public_path.write_text(public_pem)
    return KeyPair(private_pem, public_pem, private_path, public_path)


@pytest.fixture
def other_keys(other_rsa_pem: tuple[str, str], tmp_path: Path) -> KeyPair:
    """A second RSA pair, written beside the first."""
    private_pem, public_pem = other_rsa_pem
    private_path = tmp_path / "other-private.pem"
    public_path = tmp_path / "other-public.pem"
    private_path.write_text(private_pem)
    public_path.write_text(public_pem)
    return KeyPair(private_pem, public_pem, private_path, public_path)


@pytest.fixture(scope="session")
def ec_private_pem() -> str:
    """An ES256 private key, which is what Apple's client secret is signed with."""
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


@pytest.fixture(scope="session")
def ec_public_pem(ec_private_pem: str) -> str:
    """The matching ES256 public key, for verifying what was signed."""
    key = serialization.load_pem_private_key(ec_private_pem.encode(), password=None)
    return (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
