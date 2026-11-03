"""OAuth/OIDC provider protocol mechanics."""

from base.services.auth.oauth.providers import (
    OAuthProfile,
    OAuthProviderConfig,
    OAuthProviderRegistry,
    build_apple_client_secret,
    load_oauth_provider_configs,
)

__all__ = [
    "OAuthProfile",
    "OAuthProviderConfig",
    "OAuthProviderRegistry",
    "build_apple_client_secret",
    "load_oauth_provider_configs",
]
