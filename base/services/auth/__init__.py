import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi.security import HTTPAuthorizationCredentials
from jwt import (
    DecodeError,
    ExpiredSignatureError,
    InvalidAlgorithmError,
    PyJWKClient,
    decode,
    encode,
)
from jwt.exceptions import PyJWKClientError, PyJWKError

from base.models.definitions.errors import AuthErrors
from base.models.orm.user import UserDTO
from base.models.view_models.auth import (
    AccessToken,
    RefreshToken,
    TokenPair,
)
from base.services.auth.oauth import OAuthService
from base.services.common.meta import Singleton
from base.services.configuration import settings
from base.services.http import HttpRequestService
from base.services.store.user import UserStore
from base.tools.errors import raise_error


class AuthService(metaclass=Singleton):
    def __init__(self) -> None:
        self._user_store: UserStore = UserStore()
        self._oauth_service: OAuthService = OAuthService()
        self._private_key: str = self._load_key(settings.auth_private_key_path)
        self._public_key: str = self._load_key(settings.auth_public_key_path)
        self._access_token_expiry: int = settings.access_token_expiry_seconds
        self._refresh_token_expiry: int = settings.refresh_token_expiry_seconds
        self._jwks_client: PyJWKClient | None = None
        self._idp_algorithms: list[str] = [
            a.strip() for a in settings.idp_algorithms.split(",") if a.strip()
        ] or ["RS256"]

        if settings.idp_resource_server_enabled:
            self._init_idp_resource_server()

    def _init_idp_resource_server(self) -> None:
        if not settings.idp_issuer or not settings.idp_audience:
            raise ValueError(
                "IDP_ISSUER and IDP_AUDIENCE are required when "
                "IDP_RESOURCE_SERVER_ENABLED=true"
            )
        jwks_url = settings.idp_jwks_url
        if not jwks_url and settings.idp_discovery_url:
            jwks_url = self._resolve_jwks_url(settings.idp_discovery_url)
        if not jwks_url and settings.idp_issuer:
            discovery = (
                settings.idp_issuer.rstrip("/") + "/.well-known/openid-configuration"
            )
            jwks_url = self._resolve_jwks_url(discovery)
        if not jwks_url:
            raise ValueError(
                "IDP_JWKS_URL or IDP_DISCOVERY_URL (or discoverable issuer) "
                "is required when IDP_RESOURCE_SERVER_ENABLED=true"
            )
        self._jwks_client = PyJWKClient(jwks_url)

    @staticmethod
    def _resolve_jwks_url(discovery_url: str) -> str:
        resp = HttpRequestService().call_endpoint(discovery_url, method="GET")
        if resp.failed or not isinstance(resp.data, dict):
            raise ValueError(f"Failed to load IdP discovery document: {discovery_url}")
        jwks_uri = resp.data.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri:
            raise ValueError("IdP discovery document missing jwks_uri")
        return jwks_uri

    # Public

    def generate_token_pair(self, user: UserDTO) -> TokenPair:
        access_token = self._generate_access_token(
            payload={"user_id": str(user.id), "email": user.email, "type": "access"},
        )
        refresh_token = self._generate_refresh_token(
            payload={"user_id": str(user.id), "type": "refresh"},
        )
        return TokenPair(access_token=access_token, refresh_token=refresh_token)

    async def arefresh_access_token(self, refresh_token: RefreshToken) -> TokenPair:
        payload = self._decode_local_token(refresh_token, expected_type="refresh")
        user_id: str = payload.get("user_id", "")
        user = await self._user_store.aget(uuid.UUID(user_id))
        return self.generate_token_pair(user)

    async def avalidate_user(
        self,
        credentials: HTTPAuthorizationCredentials,
    ) -> UserDTO:
        token = credentials.credentials
        if settings.idp_resource_server_enabled and self._is_idp_token(token):
            return await self._avalidate_idp_user(token)
        return await self._avalidate_local_user(token)

    # Private — local JWT

    async def _avalidate_local_user(self, token: str) -> UserDTO:
        payload = self._decode_local_token(
            AccessToken(token), expected_type="access"
        )
        user_id: str | None = payload.get("user_id")
        email: str | None = payload.get("email")
        if not user_id or not email:
            raise_error(AuthErrors.USER_TOKEN_INCOMPLETE)
        return await self._user_store.aget(uuid.UUID(user_id))

    def _generate_access_token(self, payload: dict[str, Any]) -> AccessToken:
        return AccessToken(self._generate_token(payload, self._access_token_expiry))

    def _generate_refresh_token(self, payload: dict[str, Any]) -> RefreshToken:
        return RefreshToken(self._generate_token(payload, self._refresh_token_expiry))

    def _generate_token(self, payload: dict[str, Any], expiry: int) -> str:
        now = datetime.now(UTC)
        return encode(
            payload={
                **payload,
                "iat": now,
                "exp": now.timestamp() + expiry,
            },
            key=self._private_key,
            algorithm="RS256",
        )

    def _decode_local_token(
        self, token: AccessToken | RefreshToken, expected_type: str
    ) -> dict[str, Any]:
        try:
            payload: dict[str, Any] = decode(
                token,
                key=self._public_key,
                algorithms=["RS256"],
            )
            if payload.get("type") != expected_type:
                raise_error(AuthErrors.USER_TOKEN_INCOMPLETE)
            return payload
        except (DecodeError, InvalidAlgorithmError):
            raise_error(AuthErrors.USER_DECODE_FAILED)
        except ExpiredSignatureError:
            raise_error(AuthErrors.USER_TOKEN_EXPIRED)

    # Private — IdP resource server

    def _is_idp_token(self, token: str) -> bool:
        try:
            unverified: dict[str, Any] = decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
            )
        except DecodeError:
            return False
        iss = unverified.get("iss")
        return isinstance(iss, str) and iss == settings.idp_issuer

    async def _avalidate_idp_user(self, token: str) -> UserDTO:
        if self._jwks_client is None:
            raise_error(AuthErrors.IDP_NOT_CONFIGURED)
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            payload: dict[str, Any] = decode(
                token,
                key=signing_key.key,
                algorithms=self._idp_algorithms,
                issuer=settings.idp_issuer,
                audience=settings.idp_audience,
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
        email = str(payload.get(settings.idp_email_claim) or "")
        if not subject or not email:
            raise_error(AuthErrors.USER_TOKEN_INCOMPLETE)

        return await self._oauth_service.aresolve_idp_user(
            provider=settings.idp_provider_name,
            subject=subject,
            email=email,
            email_verified=bool(payload.get("email_verified", True)),
            first_name=payload.get("given_name"),
            last_name=payload.get("family_name"),
        )

    # Helpers

    @staticmethod
    def _load_key(path: str) -> str:
        with open(path) as f:
            return f.read()
