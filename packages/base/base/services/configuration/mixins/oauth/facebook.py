from pydantic import BaseModel


class FacebookOAuthSettingsMixin(BaseModel):
    oauth_facebook_client_id: str = ""
    oauth_facebook_client_secret: str = ""
    oauth_facebook_authorize_url: str = "https://www.facebook.com/v21.0/dialog/oauth"
    oauth_facebook_token_url: str = (
        "https://graph.facebook.com/v21.0/oauth/access_token"
    )
    oauth_facebook_userinfo_url: str = (
        "https://graph.facebook.com/me?fields=id,email,first_name,last_name"
    )
    oauth_facebook_scopes: str = "email public_profile"
    oauth_facebook_redirect_uri: str = ""
