from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsAppleOAuthConfig(Protocol):
    """Satisfied by `AppleOAuthSettingsMixin`."""

    oauth_apple_client_id: str
    oauth_apple_client_secret: str
    oauth_apple_team_id: str
    oauth_apple_key_id: str
    oauth_apple_private_key: str
    oauth_apple_private_key_path: str
    oauth_apple_discovery_url: str
    oauth_apple_scopes: str
    oauth_apple_redirect_uri: str
