from pydantic import BaseModel


class GoogleOAuthSettingsMixin(BaseModel):
    oauth_google_client_id: str = ""
    oauth_google_client_secret: str = ""
    oauth_google_discovery_url: str = (
        "https://accounts.google.com/.well-known/openid-configuration"
    )
    oauth_google_scopes: str = "openid email profile"
    oauth_google_redirect_uri: str = ""
