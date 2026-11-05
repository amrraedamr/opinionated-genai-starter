"""Reading a service's settings into the providers it actually offers.

Only the providers `oauth_providers` names are read, so a service composes a
settings mixin per provider it offers and declares no credentials for the rest.
"""

from typing import Any, cast

from base.models.definitions.errors import AuthErrors
from base.models.view_models.oauth import OAuthProviderConfig
from base.services.auth.oauth.apple_secret import (
    load_apple_private_key,
    resolve_apple_client_secret,
)
from base.services.configuration import get_settings
from base.services.configuration.protocols.oauth import SupportsOAuthConfig
from base.services.configuration.protocols.oauth.apple import SupportsAppleOAuthConfig
from base.services.configuration.protocols.oauth.facebook import (
    SupportsFacebookOAuthConfig,
)
from base.services.configuration.protocols.oauth.github import SupportsGitHubOAuthConfig
from base.services.configuration.protocols.oauth.google import SupportsGoogleOAuthConfig
from base.services.configuration.protocols.oauth.microsoft import (
    SupportsMicrosoftOAuthConfig,
)
from base.tools.errors import MissingSettingsMixinError, raise_error


def _config() -> SupportsOAuthConfig:
    """The settings every OAuth service carries, whatever providers it offers."""
    return cast(SupportsOAuthConfig, get_settings())


def _split_providers(raw: str) -> list[str]:
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _google(config: SupportsGoogleOAuthConfig) -> OAuthProviderConfig:
    return OAuthProviderConfig(
        name="google",
        client_id=config.oauth_google_client_id,
        client_secret=config.oauth_google_client_secret,
        redirect_uri=config.oauth_google_redirect_uri,
        scopes=config.oauth_google_scopes,
        discovery_url=config.oauth_google_discovery_url,
        profile_style="oidc",
    )


def _facebook(config: SupportsFacebookOAuthConfig) -> OAuthProviderConfig:
    return OAuthProviderConfig(
        name="facebook",
        client_id=config.oauth_facebook_client_id,
        client_secret=config.oauth_facebook_client_secret,
        redirect_uri=config.oauth_facebook_redirect_uri,
        scopes=config.oauth_facebook_scopes,
        authorize_url=config.oauth_facebook_authorize_url,
        token_url=config.oauth_facebook_token_url,
        userinfo_url=config.oauth_facebook_userinfo_url,
        profile_style="facebook",
    )


def _microsoft(config: SupportsMicrosoftOAuthConfig) -> OAuthProviderConfig:
    return OAuthProviderConfig(
        name="microsoft",
        client_id=config.oauth_microsoft_client_id,
        client_secret=config.oauth_microsoft_client_secret,
        redirect_uri=config.oauth_microsoft_redirect_uri,
        scopes=config.oauth_microsoft_scopes,
        discovery_url=config.oauth_microsoft_discovery_url,
        profile_style="oidc",
    )


def _github(config: SupportsGitHubOAuthConfig) -> OAuthProviderConfig:
    return OAuthProviderConfig(
        name="github",
        client_id=config.oauth_github_client_id,
        client_secret=config.oauth_github_client_secret,
        redirect_uri=config.oauth_github_redirect_uri,
        scopes=config.oauth_github_scopes,
        authorize_url=config.oauth_github_authorize_url,
        token_url=config.oauth_github_token_url,
        userinfo_url=config.oauth_github_userinfo_url,
        profile_style="github",
    )


def _apple(config: SupportsAppleOAuthConfig) -> OAuthProviderConfig:
    return OAuthProviderConfig(
        name="apple",
        client_id=config.oauth_apple_client_id,
        client_secret=resolve_apple_client_secret(config),
        redirect_uri=config.oauth_apple_redirect_uri,
        scopes=config.oauth_apple_scopes,
        discovery_url=config.oauth_apple_discovery_url,
        profile_style="apple",
        response_mode="form_post",
        apple_team_id=config.oauth_apple_team_id,
        apple_key_id=config.oauth_apple_key_id,
        apple_private_key=load_apple_private_key(config),
    )


_PROVIDERS: dict[str, tuple[type, str, Any]] = {
    "google": (SupportsGoogleOAuthConfig, "GoogleOAuthSettingsMixin", _google),
    "facebook": (SupportsFacebookOAuthConfig, "FacebookOAuthSettingsMixin", _facebook),
    "microsoft": (
        SupportsMicrosoftOAuthConfig,
        "MicrosoftOAuthSettingsMixin",
        _microsoft,
    ),
    "github": (SupportsGitHubOAuthConfig, "GitHubOAuthSettingsMixin", _github),
    "apple": (SupportsAppleOAuthConfig, "AppleOAuthSettingsMixin", _apple),
}
"""Every provider this package can speak to, and what a service must compose
to enable one. Order decides nothing; `oauth_providers` does."""


def _usable(cfg: OAuthProviderConfig) -> bool:
    """Whether a provider was given enough to attempt a login."""
    if not cfg.client_id or not cfg.redirect_uri:
        return False
    if cfg.name == "apple":
        can_mint = bool(cfg.apple_team_id and cfg.apple_key_id and cfg.apple_private_key)
        return bool(cfg.client_secret) or can_mint
    return bool(cfg.client_secret)


def load_oauth_provider_configs() -> dict[str, OAuthProviderConfig]:
    """The providers this service both enabled and configured.

    Only the enabled ones are read, so a service composes a mixin per provider
    it offers and declares no credentials for the rest. Enabling one whose
    mixin is missing is a configuration mistake and says so.
    """
    settings = get_settings()
    configs: dict[str, OAuthProviderConfig] = {}

    for name in _split_providers(_config().oauth_providers):
        entry = _PROVIDERS.get(name)
        if entry is None:
            raise_error(AuthErrors.OAUTH_PROVIDER_DISABLED)
        protocol, mixin, build = entry
        if not isinstance(settings, protocol):
            raise MissingSettingsMixinError(name, mixin, "OAUTH_PROVIDERS")
        cfg = build(settings)
        if _usable(cfg):
            configs[name] = cfg
    return configs
