from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsGoogleOAuthConfig(Protocol):
    """Satisfied by `GoogleOAuthSettingsMixin`."""

    oauth_google_client_id: str
    oauth_google_client_secret: str
    oauth_google_discovery_url: str
    oauth_google_scopes: str
    oauth_google_redirect_uri: str
