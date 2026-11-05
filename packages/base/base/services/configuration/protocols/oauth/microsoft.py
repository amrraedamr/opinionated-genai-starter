from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsMicrosoftOAuthConfig(Protocol):
    """Satisfied by `MicrosoftOAuthSettingsMixin`."""

    oauth_microsoft_client_id: str
    oauth_microsoft_client_secret: str
    oauth_microsoft_discovery_url: str
    oauth_microsoft_scopes: str
    oauth_microsoft_redirect_uri: str
