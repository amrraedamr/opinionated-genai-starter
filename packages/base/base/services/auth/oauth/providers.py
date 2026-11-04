import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode

from jwt import encode
from pydantic import ValidationError

from base.models.definitions.errors import AuthErrors
from base.services.auth.models import TokenResponse
from base.services.configuration import get_settings
from base.services.configuration.mixins import OAuthSettingsMixin
from base.services.http import HttpRequestService
from base.tools.errors import raise_error


def _config() -> OAuthSettingsMixin:
    """The service's OAuth settings."""
    return cast(OAuthSettingsMixin, get_settings())


@dataclass(frozen=True, slots=True)
class OAuthProviderConfig:
    name: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: str
    discovery_url: str = ""
    authorize_url: str = ""
    token_url: str = ""
    userinfo_url: str = ""
    # oidc | facebook | github | apple
    profile_style: str = "oidc"
    response_mode: str = ""
    # Apple ES256 client-secret material (secret regenerated at token exchange)
    apple_team_id: str = ""
    apple_key_id: str = ""
    apple_private_key: str = ""


@dataclass(frozen=True, slots=True)
class OAuthProfile:
    subject: str
    email: str
    email_verified: bool
    first_name: str | None = None
    last_name: str | None = None


def _split_providers(raw: str) -> list[str]:
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _load_apple_private_key() -> str:
    raw = _config().oauth_apple_private_key.strip()
    if raw:
        return raw.replace("\\n", "\n")
    path = _config().oauth_apple_private_key_path.strip()
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


def _resolve_apple_client_secret() -> str:
    if _config().oauth_apple_client_secret:
        return _config().oauth_apple_client_secret
    private_key = _load_apple_private_key()
    if not (
        _config().oauth_apple_client_id
        and _config().oauth_apple_team_id
        and _config().oauth_apple_key_id
        and private_key
    ):
        return ""
    return build_apple_client_secret(
        client_id=_config().oauth_apple_client_id,
        team_id=_config().oauth_apple_team_id,
        key_id=_config().oauth_apple_key_id,
        private_key_pem=private_key,
    )


def load_oauth_provider_configs() -> dict[str, OAuthProviderConfig]:
    enabled = set(_split_providers(_config().oauth_providers))
    configs: dict[str, OAuthProviderConfig] = {}

    google = OAuthProviderConfig(
        name="google",
        client_id=_config().oauth_google_client_id,
        client_secret=_config().oauth_google_client_secret,
        redirect_uri=_config().oauth_google_redirect_uri,
        scopes=_config().oauth_google_scopes,
        discovery_url=_config().oauth_google_discovery_url,
        profile_style="oidc",
    )
    facebook = OAuthProviderConfig(
        name="facebook",
        client_id=_config().oauth_facebook_client_id,
        client_secret=_config().oauth_facebook_client_secret,
        redirect_uri=_config().oauth_facebook_redirect_uri,
        scopes=_config().oauth_facebook_scopes,
        authorize_url=_config().oauth_facebook_authorize_url,
        token_url=_config().oauth_facebook_token_url,
        userinfo_url=_config().oauth_facebook_userinfo_url,
        profile_style="facebook",
    )
    microsoft = OAuthProviderConfig(
        name="microsoft",
        client_id=_config().oauth_microsoft_client_id,
        client_secret=_config().oauth_microsoft_client_secret,
        redirect_uri=_config().oauth_microsoft_redirect_uri,
        scopes=_config().oauth_microsoft_scopes,
        discovery_url=_config().oauth_microsoft_discovery_url,
        profile_style="oidc",
    )
    github = OAuthProviderConfig(
        name="github",
        client_id=_config().oauth_github_client_id,
        client_secret=_config().oauth_github_client_secret,
        redirect_uri=_config().oauth_github_redirect_uri,
        scopes=_config().oauth_github_scopes,
        authorize_url=_config().oauth_github_authorize_url,
        token_url=_config().oauth_github_token_url,
        userinfo_url=_config().oauth_github_userinfo_url,
        profile_style="github",
    )
    apple_secret = _resolve_apple_client_secret()
    apple = OAuthProviderConfig(
        name="apple",
        client_id=_config().oauth_apple_client_id,
        client_secret=apple_secret,
        redirect_uri=_config().oauth_apple_redirect_uri,
        scopes=_config().oauth_apple_scopes,
        discovery_url=_config().oauth_apple_discovery_url,
        profile_style="apple",
        response_mode="form_post",
        apple_team_id=_config().oauth_apple_team_id,
        apple_key_id=_config().oauth_apple_key_id,
        apple_private_key=_load_apple_private_key(),
    )

    for cfg in (google, facebook, microsoft, github, apple):
        if cfg.name not in enabled:
            continue
        if not cfg.client_id or not cfg.redirect_uri:
            continue
        if cfg.name == "apple":
            can_mint = bool(
                cfg.apple_team_id and cfg.apple_key_id and cfg.apple_private_key
            )
            if not cfg.client_secret and not can_mint:
                continue
        elif not cfg.client_secret:
            continue
        configs[cfg.name] = cfg
    return configs


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

        from jwt import decode

        claims = decode(id_token, options={"verify_signature": False})
        subject = str(claims.get("sub") or "")
        email = str(claims.get("email") or "")
        email_verified_raw = claims.get("email_verified", True)
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
            from jwt import decode

            claims = decode(id_token, options={"verify_signature": False})
            email = str(claims.get("email") or "")
            subject = str(claims.get("sub") or "")
            if subject and email:
                return OAuthProfile(
                    subject=subject,
                    email=email,
                    email_verified=bool(claims.get("email_verified", True)),
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
            email_verified=bool(body.get("email_verified", True)),
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
            email_verified=True,
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
        if not email:
            raw = await self._http.acall_endpoint(
                "https://api.github.com/user/emails",
                method="GET",
                headers=headers,
                raw=True,
            )
            items = json.loads(raw)
            if isinstance(items, list):
                for item in items:
                    if (
                        isinstance(item, dict)
                        and item.get("primary")
                        and item.get("verified")
                    ):
                        email = str(item.get("email") or "")
                        break
        if not subject or not email:
            raise_error(AuthErrors.OAUTH_PROFILE_INCOMPLETE)
        name = str(body.get("name") or "").split(" ", 1)
        return OAuthProfile(
            subject=subject,
            email=email,
            email_verified=True,
            first_name=name[0] or None,
            last_name=name[1] if len(name) > 1 else None,
        )
