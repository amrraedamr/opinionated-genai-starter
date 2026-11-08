import json
from typing import Any
from urllib.parse import urlencode

from pydantic import ValidationError

from base.tools.errors import MissingExtraError

try:
    from jwt import decode
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("auth", "pyjwt", feature="OAuth providers") from exc

from base.models.definitions.errors import AuthErrors
from base.models.view_models.oauth import OAuthProfile, OAuthProviderConfig
from base.services.auth.models import TokenResponse
from base.services.auth.oauth.apple_secret import build_apple_client_secret
from base.services.auth.oauth.config import load_oauth_provider_configs
from base.services.http import HttpRequestService
from base.tools.errors import raise_error


class OAuthProviderRegistry:
    def __init__(self) -> None:
        self._http = HttpRequestService()
        self._configs = load_oauth_provider_configs()
        self._discovery_cache: dict[str, dict[str, Any]] = {}

    def get(self, provider: str) -> OAuthProviderConfig:
        cfg = self._configs.get(provider.lower())
        if cfg is None:
            raise_error(AuthErrors.OAUTH_PROVIDER_DISABLED)
        return cfg

    def enabled_names(self) -> list[str]:
        return sorted(self._configs)

    def resolve_client_secret(self, cfg: OAuthProviderConfig) -> str:
        if cfg.profile_style == "apple" and cfg.apple_private_key:
            return build_apple_client_secret(
                client_id=cfg.client_id,
                team_id=cfg.apple_team_id,
                key_id=cfg.apple_key_id,
                private_key_pem=cfg.apple_private_key,
            )
        return cfg.client_secret

    async def aresolve_endpoints(
        self, cfg: OAuthProviderConfig
    ) -> tuple[str, str, str]:
        if cfg.authorize_url and cfg.token_url:
            return cfg.authorize_url, cfg.token_url, cfg.userinfo_url

        if not cfg.discovery_url:
            raise_error(AuthErrors.OAUTH_PROVIDER_DISABLED)

        cached = self._discovery_cache.get(cfg.discovery_url)
        if cached is None:
            resp = await self._http.acall_endpoint(cfg.discovery_url, method="GET")
            if resp.failed or not isinstance(resp.data, dict):
                raise_error(AuthErrors.OAUTH_EXCHANGE_FAILED)
            cached = resp.data
            self._discovery_cache[cfg.discovery_url] = cached

        authorize = str(cached.get("authorization_endpoint", ""))
        token = str(cached.get("token_endpoint", ""))
        userinfo = str(cached.get("userinfo_endpoint", cfg.userinfo_url or ""))
        if not authorize or not token:
            raise_error(AuthErrors.OAUTH_EXCHANGE_FAILED)
        return authorize, token, userinfo

    def build_authorize_url(
        self,
        authorize_url: str,
        cfg: OAuthProviderConfig,
        *,
        state: str,
        code_challenge: str,
    ) -> str:
        params = {
            "client_id": cfg.client_id,
            "redirect_uri": cfg.redirect_uri,
            "response_type": "code",
            "scope": cfg.scopes,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if cfg.response_mode:
            params["response_mode"] = cfg.response_mode
        separator = "&" if "?" in authorize_url else "?"
        return f"{authorize_url}{separator}{urlencode(params)}"

    async def aexchange_code(
        self,
        cfg: OAuthProviderConfig,
        *,
        code: str,
        code_verifier: str,
        token_url: str,
    ) -> TokenResponse:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": cfg.redirect_uri,
            "client_id": cfg.client_id,
            "client_secret": self.resolve_client_secret(cfg),
            "code_verifier": code_verifier,
        }
        headers = {"Accept": "application/json"}
        resp = await self._http.acall_endpoint(
            token_url,
            method="POST",
            data=data,
            headers=headers,
        )
        if resp.failed or not isinstance(resp.data, dict):
            raise_error(AuthErrors.OAUTH_EXCHANGE_FAILED)
        try:
            return TokenResponse.model_validate(resp.data)
        except ValidationError:
            raise_error(AuthErrors.OAUTH_EXCHANGE_FAILED)

    async def afetch_profile(
        self,
        cfg: OAuthProviderConfig,
        *,
        token_payload: TokenResponse,
        userinfo_url: str,
        apple_user_json: str | None = None,
    ) -> OAuthProfile:
        if cfg.profile_style == "oidc":
            return await self._afetch_oidc_profile(token_payload, userinfo_url)
        if cfg.profile_style == "apple":
            return self._fetch_apple_profile(token_payload, apple_user_json)
        if cfg.profile_style == "facebook":
            return await self._afetch_facebook_profile(cfg, token_payload)
        if cfg.profile_style == "github":
            return await self._afetch_github_profile(cfg, token_payload)
        raise_error(AuthErrors.OAUTH_PROVIDER_DISABLED)

    def _fetch_apple_profile(
        self,
        token_payload: TokenResponse,
        apple_user_json: str | None,
    ) -> OAuthProfile:
        id_token = token_payload.id_token
        if not id_token:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)

        claims = decode(id_token, options={"verify_signature": False})
        subject = str(claims.get("sub") or "")
        email = str(claims.get("email") or "")
        email_verified_raw = claims.get("email_verified", False)
        email_verified = (
            email_verified_raw
            if isinstance(email_verified_raw, bool)
            else str(email_verified_raw).lower() == "true"
        )

        first_name: str | None = None
        last_name: str | None = None
        if apple_user_json:
            try:
                user_blob = json.loads(apple_user_json)
            except json.JSONDecodeError:
                user_blob = None
            if isinstance(user_blob, dict):
                if not email:
                    email = str(user_blob.get("email") or "")
                name = user_blob.get("name")
                if isinstance(name, dict):
                    first_name = name.get("firstName")
                    last_name = name.get("lastName")

        if not subject or not email:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        return OAuthProfile(
            subject=subject,
            email=email,
            email_verified=email_verified,
            first_name=first_name,
            last_name=last_name,
        )

    async def _afetch_oidc_profile(
        self, token_payload: TokenResponse, userinfo_url: str
    ) -> OAuthProfile:
        id_token = token_payload.id_token
        if id_token:
            claims = decode(id_token, options={"verify_signature": False})
            email = str(claims.get("email") or "")
            subject = str(claims.get("sub") or "")
            if subject and email:
                return OAuthProfile(
                    subject=subject,
                    email=email,
                    email_verified=bool(claims.get("email_verified", False)),
                    first_name=claims.get("given_name"),
                    last_name=claims.get("family_name"),
                )

        access_token = token_payload.access_token
        if not userinfo_url:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)

        resp = await self._http.acall_endpoint(
            userinfo_url,
            method="GET",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if resp.failed or not isinstance(resp.data, dict):
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        body = resp.data
        email = str(body.get("email") or "")
        subject = str(body.get("sub") or "")
        if not subject or not email:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        return OAuthProfile(
            subject=subject,
            email=email,
            email_verified=bool(body.get("email_verified", False)),
            first_name=body.get("given_name"),
            last_name=body.get("family_name"),
        )

    async def _afetch_facebook_profile(
        self, cfg: OAuthProviderConfig, token_payload: TokenResponse
    ) -> OAuthProfile:
        access_token = token_payload.access_token
        if not cfg.userinfo_url:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        separator = "&" if "?" in cfg.userinfo_url else "?"
        url = (
            f"{cfg.userinfo_url}{separator}{urlencode({'access_token': access_token})}"
        )
        resp = await self._http.acall_endpoint(url, method="GET")
        if resp.failed or not isinstance(resp.data, dict):
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        body = resp.data
        subject = str(body.get("id") or "")
        email = str(body.get("email") or "")
        if not subject or not email:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        return OAuthProfile(
            subject=subject,
            email=email,
            # Facebook's profile endpoint reports no verification state.
            email_verified=False,
            first_name=body.get("first_name"),
            last_name=body.get("last_name"),
        )

    async def _afetch_github_profile(
        self, cfg: OAuthProviderConfig, token_payload: TokenResponse
    ) -> OAuthProfile:
        access_token = token_payload.access_token
        if not cfg.userinfo_url:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        resp = await self._http.acall_endpoint(
            cfg.userinfo_url, method="GET", headers=headers
        )
        if resp.failed or not isinstance(resp.data, dict):
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        body = resp.data
        subject = str(body.get("id") or "")
        email = str(body.get("email") or "")

        # The profile endpoint returns whichever address the user chose to
        # publish, verified or not, and says which only on /user/emails. The
        # address is read from there whenever one is offered, so a published
        # address is trusted exactly when GitHub itself vouches for it.
        verified_primary = await self._agithub_verified_primary(headers)
        if verified_primary and (not email or verified_primary == email):
            email, email_verified = verified_primary, True
        else:
            email_verified = False

        if not subject or not email:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        name = str(body.get("name") or "").split(" ", 1)
        return OAuthProfile(
            subject=subject,
            email=email,
            email_verified=email_verified,
            first_name=name[0] or None,
            last_name=name[1] if len(name) > 1 else None,
        )

    async def _agithub_verified_primary(self, headers: dict[str, str]) -> str:
        """GitHub's primary address, only when GitHub reports it verified."""
        raw = await self._http.acall_endpoint(
            "https://api.github.com/user/emails",
            method="GET",
            headers=headers,
            raw=True,
        )
        try:
            items = json.loads(raw)
        except (TypeError, ValueError):
            return ""
        if not isinstance(items, list):
            return ""
        for item in items:
            if isinstance(item, dict) and item.get("primary") and item.get("verified"):
                return str(item.get("email") or "")
        return ""
