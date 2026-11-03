from pydantic import BaseModel


class OAuthSettingsMixin(BaseModel):
    """Login-with-X. `oauth_providers` is a comma-separated list of enabled names."""

    oauth_providers: str = ""
    oauth_success_redirect: str = "http://localhost:3000/auth/callback"
    oauth_state_ttl_seconds: int = 600
    oauth_exchange_code_ttl_seconds: int = 60

    oauth_google_client_id: str = ""
    oauth_google_client_secret: str = ""
    oauth_google_discovery_url: str = (
        "https://accounts.google.com/.well-known/openid-configuration"
    )
    oauth_google_scopes: str = "openid email profile"
    oauth_google_redirect_uri: str = ""

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

    oauth_microsoft_client_id: str = ""
    oauth_microsoft_client_secret: str = ""
    oauth_microsoft_discovery_url: str = (
        "https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration"
    )
    oauth_microsoft_scopes: str = "openid email profile"
    oauth_microsoft_redirect_uri: str = ""

    oauth_github_client_id: str = ""
    oauth_github_client_secret: str = ""
    oauth_github_authorize_url: str = "https://github.com/login/oauth/authorize"
    oauth_github_token_url: str = "https://github.com/login/oauth/access_token"
    oauth_github_userinfo_url: str = "https://api.github.com/user"
    oauth_github_scopes: str = "read:user user:email"
    oauth_github_redirect_uri: str = ""

    # Sign in with Apple — client_secret is a JWT: set TEAM_ID + KEY_ID + private
    # key (path or PEM), or paste a pre-generated OAUTH_APPLE_CLIENT_SECRET.
    oauth_apple_client_id: str = ""
    oauth_apple_client_secret: str = ""
    oauth_apple_team_id: str = ""
    oauth_apple_key_id: str = ""
    oauth_apple_private_key: str = ""
    oauth_apple_private_key_path: str = ""
    oauth_apple_discovery_url: str = (
        "https://appleid.apple.com/.well-known/openid-configuration"
    )
    oauth_apple_scopes: str = "name email"
    oauth_apple_redirect_uri: str = ""
