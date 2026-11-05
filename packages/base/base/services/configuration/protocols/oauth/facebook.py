from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsFacebookOAuthConfig(Protocol):
    """Satisfied by `FacebookOAuthSettingsMixin`."""

    oauth_facebook_client_id: str
    oauth_facebook_client_secret: str
    oauth_facebook_authorize_url: str
    oauth_facebook_token_url: str
    oauth_facebook_userinfo_url: str
    oauth_facebook_scopes: str
    oauth_facebook_redirect_uri: str
