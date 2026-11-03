"""User-bound authentication."""

import uuid

from base.models.definitions.errors import AuthErrors
from base.services.auth import Principal, TokenIssuer, TokenPair, TokenVerifier
from base.services.common.meta import Singleton
from base.tools.errors import raise_error
from identity.models.orm.user import UserDTO
from identity.models.view_models.auth import RefreshToken
from identity.services.auth.oauth import OAuthService
from identity.settings import settings
from identity.stores.user import UserStore


class AuthService(metaclass=Singleton):
    def __init__(self) -> None:
        self._user_store: UserStore = UserStore()
        self._oauth_service: OAuthService = OAuthService()
        self._issuer: TokenIssuer = TokenIssuer()
        self._verifier: TokenVerifier = TokenVerifier()

    # Issuing

    def generate_token_pair(self, user: UserDTO) -> TokenPair:
        """Issue an access/refresh pair for a user."""
        return self._issuer.issue_pair(
            {"user_id": str(user.id), "email": user.email},
        )

    async def arefresh_access_token(self, refresh_token: RefreshToken) -> TokenPair:
        payload = self._verifier.decode_local(refresh_token, expected_type="refresh")
        user_id = payload.get("user_id", "")
        user = await self._user_store.aget(uuid.UUID(user_id))
        return self.generate_token_pair(user)

    # Resolving

    async def aresolve_principal(self, principal: Principal) -> UserDTO:
        """Resolve a verified principal to this service's user."""
        if self._is_idp_principal(principal):
            return await self._aresolve_idp_principal(principal)
        if principal.user_id is None:
            raise_error(AuthErrors.USER_TOKEN_INCOMPLETE)
        return await self._user_store.aget(principal.user_id)

    def _is_idp_principal(self, principal: Principal) -> bool:
        return bool(
            settings.idp_resource_server_enabled
            and principal.issuer
            and principal.issuer == settings.idp_issuer
        )

    async def _aresolve_idp_principal(self, principal: Principal) -> UserDTO:
        """Link or create the local user for an IdP principal."""
        if not principal.email:
            raise_error(AuthErrors.USER_TOKEN_INCOMPLETE)
        claims = principal.claims
        return await self._oauth_service.aresolve_idp_user(
            provider=settings.idp_provider_name,
            subject=principal.subject,
            email=principal.email,
            email_verified=bool(claims.get("email_verified", True)),
            first_name=claims.get("given_name"),
            last_name=claims.get("family_name"),
        )
