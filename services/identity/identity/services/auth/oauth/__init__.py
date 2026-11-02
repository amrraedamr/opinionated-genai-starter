from typing import Any
from urllib.parse import urlencode

from base.models.definitions.errors import AuthErrors
from base.services.auth import TokenIssuer, TokenPair, TokenVerifier
from base.services.auth.oauth import OAuthProfile, OAuthProviderRegistry
from base.services.auth.tokens import InvalidSignedPayload
from base.services.common.meta import Singleton
from base.tools.errors import raise_error
from base.tools.security import pkce_pair
from identity.models.orm.user import UserDTO
from identity.models.orm.user_identity import UserIdentityDTO
from identity.settings import settings
from identity.stores.user import UserStore
from identity.stores.user_identity import UserIdentityStore


class OAuthService(metaclass=Singleton):
    def __init__(self) -> None:
        self._registry = OAuthProviderRegistry()
        self._user_store = UserStore()
        self._identity_store = UserIdentityStore()
        self._issuer = TokenIssuer()
        self._verifier = TokenVerifier()

    def enabled_providers(self) -> list[str]:
        return self._registry.enabled_names()

    async def abuild_authorize_redirect(self, provider: str) -> str:
        cfg = self._registry.get(provider)
        authorize_url, _, _ = await self._registry.aresolve_endpoints(cfg)
        code_verifier, code_challenge = pkce_pair()
        state = self._sign_state(
            {
                "type": "oauth_state",
                "provider": cfg.name,
                "code_verifier": code_verifier,
            },
            settings.oauth_state_ttl_seconds,
        )
        return self._registry.build_authorize_url(
            authorize_url,
            cfg,
            state=state,
            code_challenge=code_challenge,
        )

    async def ahandle_callback(
        self,
        provider: str,
        *,
        code: str,
        state: str,
        apple_user_json: str | None = None,
    ) -> str:
        """Complete OAuth; return opaque exchange code for the SPA."""
        cfg = self._registry.get(provider)
        state_payload = self._decode_signed(state, expected_type="oauth_state")
        if state_payload.get("provider") != cfg.name:
            raise_error(AuthErrors.OAUTH_STATE_INVALID)
        code_verifier = str(state_payload.get("code_verifier") or "")
        if not code_verifier:
            raise_error(AuthErrors.OAUTH_STATE_INVALID)

        _, token_url, userinfo_url = await self._registry.aresolve_endpoints(cfg)
        token_payload = await self._registry.aexchange_code(
            cfg,
            code=code,
            code_verifier=code_verifier,
            token_url=token_url,
        )
        profile = await self._registry.afetch_profile(
            cfg,
            token_payload=token_payload,
            userinfo_url=userinfo_url,
            apple_user_json=apple_user_json,
        )
        user = await self.aresolve_oauth_user(cfg.name, profile)
        # ponytail: short-lived signed code, not single-use across workers;
        # upgrade to Redis/DB jti denylist if replay window matters.
        return self._sign_state(
            {
                "type": "oauth_code",
                "user_id": str(user.id),
            },
            settings.oauth_exchange_code_ttl_seconds,
        )

    def success_redirect_url(self, exchange_code: str) -> str:
        base = settings.oauth_success_redirect
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}{urlencode({'code': exchange_code})}"

    async def aexchange_code(self, code: str) -> TokenPair:
        from identity.services.auth import AuthService

        payload = self._decode_signed(code, expected_type="oauth_code")
        user_id = str(payload.get("user_id") or "")
        if not user_id:
            raise_error(AuthErrors.OAUTH_CODE_INVALID)
        user = await self._user_store.aget(user_id)
        return AuthService().generate_token_pair(user)

    async def aresolve_oauth_user(
        self, provider: str, profile: OAuthProfile
    ) -> UserDTO:
        if not profile.subject or not profile.email:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)

        identity = await self._identity_store.aget_by_provider_subject(
            provider, profile.subject
        )
        if identity is not None:
            return await self._user_store.aget(identity.user_id)

        existing: UserDTO | None = None
        if profile.email_verified:
            existing = await self._user_store._aget(profile.email, column_name="email")

        if existing is None:
            existing = await self._user_store.aadd(
                UserDTO(
                    email=profile.email,
                    password_hash=None,
                    first_name=profile.first_name,
                    last_name=profile.last_name,
                )
            )

        await self._identity_store.aadd(
            UserIdentityDTO(
                user_id=existing.id,
                provider=provider,
                subject=profile.subject,
                email_at_link=profile.email,
            )
        )
        return existing

    async def aresolve_idp_user(
        self,
        *,
        provider: str,
        subject: str,
        email: str,
        email_verified: bool = True,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> UserDTO:
        return await self.aresolve_oauth_user(
            provider,
            OAuthProfile(
                subject=subject,
                email=email,
                email_verified=email_verified,
                first_name=first_name,
                last_name=last_name,
            ),
        )

    def _sign_state(self, payload: dict[str, Any], ttl_seconds: int) -> str:
        return self._issuer.sign_payload(payload, ttl_seconds=ttl_seconds)

    def _decode_signed(self, token: str, expected_type: str) -> dict[str, Any]:
        invalid = (
            AuthErrors.OAUTH_CODE_INVALID
            if expected_type == "oauth_code"
            else AuthErrors.OAUTH_STATE_INVALID
        )
        try:
            payload = self._verifier.decode_payload(token)
        except InvalidSignedPayload:
            raise_error(invalid)
        if payload.get("type") != expected_type:
            raise_error(invalid)
        return payload
