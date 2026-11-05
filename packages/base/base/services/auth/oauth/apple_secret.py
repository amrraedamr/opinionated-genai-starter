"""Minting Apple's client secret, which is a signed assertion rather than a string.

Sign in with Apple does not issue a static client secret. One is built from the
team id, key id and private key, signed ES256, and expires — so it is produced
per exchange rather than stored.
"""

import time

from base.tools.errors import MissingExtraError

try:
    from jwt import encode
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("auth", "pyjwt", feature="Sign in with Apple") from exc

from pathlib import Path

from base.services.configuration.protocols import (
    SupportsAppleOAuthConfig,
)


def load_apple_private_key(config: SupportsAppleOAuthConfig) -> str:
    raw = config.oauth_apple_private_key.strip()
    if raw:
        return raw.replace("\\n", "\n")
    path = config.oauth_apple_private_key_path.strip()
    if path:
        return Path(path).read_text(encoding="utf-8")
    return ""


def build_apple_client_secret(
    *,
    client_id: str,
    team_id: str,
    key_id: str,
    private_key_pem: str,
    ttl_seconds: int = 86400,
) -> str:
    now = int(time.time())
    return encode(
        {
            "iss": team_id,
            "iat": now,
            "exp": now + ttl_seconds,
            "aud": "https://appleid.apple.com",
            "sub": client_id,
        },
        private_key_pem,
        algorithm="ES256",
        headers={"kid": key_id, "alg": "ES256"},
    )


def resolve_apple_client_secret(config: SupportsAppleOAuthConfig) -> str:
    if config.oauth_apple_client_secret:
        return config.oauth_apple_client_secret
    private_key = load_apple_private_key(config)
    if not (
        config.oauth_apple_client_id
        and config.oauth_apple_team_id
        and config.oauth_apple_key_id
        and private_key
    ):
        return ""
    return build_apple_client_secret(
        client_id=config.oauth_apple_client_id,
        team_id=config.oauth_apple_team_id,
        key_id=config.oauth_apple_key_id,
        private_key_pem=private_key,
    )
