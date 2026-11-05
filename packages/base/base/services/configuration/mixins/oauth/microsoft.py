from pydantic import BaseModel


class MicrosoftOAuthSettingsMixin(BaseModel):
    oauth_microsoft_client_id: str = ""
    oauth_microsoft_client_secret: str = ""
    oauth_microsoft_discovery_url: str = (
        "https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration"
    )
    oauth_microsoft_scopes: str = "openid email profile"
    oauth_microsoft_redirect_uri: str = ""
