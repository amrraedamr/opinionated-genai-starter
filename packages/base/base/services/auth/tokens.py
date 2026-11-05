"""JWT signing and verification."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from base.models.definitions.errors import AuthErrors
from base.services.auth.models import AccessToken, Principal, RefreshToken, TokenPair
from base.services.common.meta import Singleton
from base.services.configuration import settings_for
from base.services.configuration.protocols import (
    SupportsIdpConfig,
    SupportsTokenIssuingConfig,
    SupportsTokenVerificationConfig,
)
from base.tools.errors import InvalidSignedPayload, MissingExtraError, raise_error
from base.tools.misc import parse_uuid, split_csv

try:
    from jwt import (
        DecodeError,
        ExpiredSignatureError,
        InvalidAlgorithmError,
        PyJWKClient,
        decode,
        encode,
    )
    from jwt.exceptions import PyJWKClientError, PyJWKError
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "auth", "pyjwt", feature="token issuing/verification"
    ) from exc


class TokenIssuer(metaclass=Singleton):
    """Signs first-party RS256 access and refresh tokens."""

    def __init__(self, config: SupportsTokenIssuingConfig | None = None) -> None:
        self._config = config or settings_for(SupportsTokenIssuingConfig)
        self._private_key: str = Path(self._config.auth_private_key_path).read_text()

    def issue_pair(self, claims: dict[str, Any]) -> TokenPair:
        """Issue an access/refresh pair around caller-supplied claims."""
        access = self.issue_access(claims)
        refresh = self.issue_refresh({k: v for k, v in claims.items() if k != "email"})
        return TokenPair(access_token=access, refresh_token=refresh)

    def issue_access(self, claims: dict[str, Any]) -> AccessToken:
        expiry = self._config.access_token_expiry_seconds
        return AccessToken(self._sign({**claims, "type": "access"}, expiry))

    def issue_refresh(self, claims: dict[str, Any]) -> RefreshToken:
        expiry = self._config.refresh_token_expiry_seconds
        return RefreshToken(self._sign({**claims, "type": "refresh"}, expiry))

    def sign_payload(self, payload: dict[str, Any], *, ttl_seconds: int) -> str:
        """Sign an arbitrary short-lived payload, such as an OAuth state."""
        return self._sign(payload, ttl_seconds)

    def _sign(self, payload: dict[str, Any], expiry_seconds: int) -> str:
        now = datetime.now(UTC)
        return encode(
            payload={
                **payload,
                "iat": now,
                "exp": now.timestamp() + expiry_seconds,
            },
            key=self._private_key,
            algorithm="RS256",
        )


class TokenVerifier(metaclass=Singleton):
    """Verifies a bearer token and returns a `Principal`."""

    def __init__(self, config: SupportsTokenVerificationConfig | None = None) -> None:
        self._config = config or settings_for(SupportsTokenVerificationConfig)
        self._public_key: str = Path(self._config.auth_public_key_path).read_text()
        self._jwks_client: PyJWKClient | None = None
        if self._idp_enabled:
            self._init_idp()

    # IdP wiring

    @property
    def _idp(self) -> SupportsIdpConfig | None:
        config = self._config
        return (
            cast(SupportsIdpConfig, config)
            if isinstance(config, SupportsIdpConfig)
            else None
        )

    @property
    def _idp_enabled(self) -> bool:
        idp = self._idp
        return bool(idp and idp.idp_resource_server_enabled)

    def _init_idp(self) -> None:
        idp = self._idp
        if idp is None:  # pragma: no cover
            return
        if not idp.idp_issuer or not idp.idp_audience:
            raise ValueError(
                "IDP_ISSUER and IDP_AUDIENCE are required when "
                "IDP_RESOURCE_SERVER_ENABLED=true"
            )
        jwks_url = idp.idp_jwks_url
        if not jwks_url and idp.idp_discovery_url:
            jwks_url = self._resolve_jwks_url(idp.idp_discovery_url)
        if not jwks_url and idp.idp_issuer:
            discovery = idp.idp_issuer.rstrip("/") + "/.well-known/openid-configuration"
            jwks_url = self._resolve_jwks_url(discovery)
        if not jwks_url:
            raise ValueError(
                "IDP_JWKS_URL or IDP_DISCOVERY_URL (or a discoverable issuer) "
                "is required when IDP_RESOURCE_SERVER_ENABLED=true"
            )
        self._jwks_client = PyJWKClient(jwks_url)

    @staticmethod
    def _resolve_jwks_url(discovery_url: str) -> str:
        from base.services.http import HttpRequestService

        response = HttpRequestService().call_endpoint(discovery_url, method="GET")
        if response.failed or not isinstance(response.data, dict):
            raise ValueError(f"Failed to load IdP discovery document: {discovery_url}")
        jwks_uri = response.data.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri:
            raise ValueError("IdP discovery document missing jwks_uri")
        return jwks_uri

    # Public

    def verify(self, token: str, *, expected_type: str = "access") -> Principal:
        """Verify any supported token and return the caller it asserts."""
        if self._idp_enabled and self._is_idp_token(token):
            return self._verify_idp(token)
        return self._verify_local(token, expected_type=expected_type)

    def decode_local(self, token: str, *, expected_type: str) -> dict[str, Any]:
        """Verify a first-party token and return its raw claims."""
        try:
            payload: dict[str, Any] = decode(
                token,
                key=self._public_key,
                algorithms=["RS256"],
            )
        except (DecodeError, InvalidAlgorithmError):
            raise_error(AuthErrors.USER_DECODE_FAILED)
        except ExpiredSignatureError:
            raise_error(AuthErrors.USER_TOKEN_EXPIRED)
        if payload.get("type") != expected_type:
            raise_error(AuthErrors.USER_TOKEN_INCOMPLETE)
        return payload

    def decode_payload(self, token: str) -> dict[str, Any]:
        """Verify an arbitrary signed payload and return its claims.

        Raises `InvalidSignedPayload` so the caller can map it to an error of
        its own, rather than imposing the token error codes used for callers.
        """
        try:
            return decode(token, key=self._public_key, algorithms=["RS256"])
        except ExpiredSignatureError as exc:
            raise InvalidSignedPayload(expired=True) from exc
        except (DecodeError, InvalidAlgorithmError) as exc:
            raise InvalidSignedPayload(expired=False) from exc

    # Local

    def _verify_local(self, token: str, *, expected_type: str) -> Principal:
        payload = self.decode_local(token, expected_type=expected_type)
        subject = str(payload.get("user_id") or payload.get("sub") or "")
        if not subject:
            raise_error(AuthErrors.USER_TOKEN_INCOMPLETE)
        return Principal(
            subject=subject,
            user_id=parse_uuid(subject),
            email=payload.get("email"),
            roles=frozenset(payload.get("roles", []) or []),
            scopes=frozenset((payload.get("scope") or "").split()),
            issuer=payload.get("iss"),
            token_type=payload.get("type"),
            claims=payload,
        )

    # IdP

    def _is_idp_token(self, token: str) -> bool:
        idp = self._idp
        if idp is None:
            return False
        try:
            unverified: dict[str, Any] = decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
            )
        except DecodeError:
            return False
        issuer = unverified.get("iss")
        return isinstance(issuer, str) and issuer == idp.idp_issuer

    def _verify_idp(self, token: str) -> Principal:
        idp = self._idp
        if self._jwks_client is None or idp is None:
            raise_error(AuthErrors.IDP_NOT_CONFIGURED)
        algorithms = split_csv(idp.idp_algorithms) or ["RS256"]
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            payload: dict[str, Any] = decode(
                token,
                key=signing_key.key,
                algorithms=algorithms,
                issuer=idp.idp_issuer,
                audience=idp.idp_audience,
            )
        except ExpiredSignatureError:
            raise_error(AuthErrors.USER_TOKEN_EXPIRED)
        except (
            DecodeError,
            InvalidAlgorithmError,
            PyJWKClientError,
            PyJWKError,
            ValueError,
        ):
            raise_error(AuthErrors.USER_DECODE_FAILED)

        subject = str(payload.get("sub") or "")
        email = str(payload.get(idp.idp_email_claim) or "")
        if not subject or not email:
            raise_error(AuthErrors.USER_TOKEN_INCOMPLETE)
        return Principal(
            subject=subject,
            user_id=parse_uuid(subject),
            email=email,
            roles=frozenset(payload.get("roles", []) or []),
            scopes=frozenset((payload.get("scope") or "").split()),
            issuer=idp.idp_issuer,
            token_type="access",
            claims=payload,
        )
