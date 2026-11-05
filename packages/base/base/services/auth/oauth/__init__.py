"""OAuth/OIDC provider protocol mechanics."""

from base.models.view_models.oauth import OAuthProfile, OAuthProviderConfig
from base.services.auth.oauth.apple_secret import build_apple_client_secret
from base.services.auth.oauth.config import load_oauth_provider_configs
from base.services.auth.oauth.providers import OAuthProviderRegistry

__all__ = [
    "OAuthProfile",
    "OAuthProviderConfig",
    "OAuthProviderRegistry",
    "build_apple_client_secret",
    "load_oauth_provider_configs",
]
