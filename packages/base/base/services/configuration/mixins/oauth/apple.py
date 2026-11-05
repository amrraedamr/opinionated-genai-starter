from pydantic import BaseModel


class AppleOAuthSettingsMixin(BaseModel):
    """Sign in with Apple.

    The client secret is a JWT. Either set the team id, key id and private key
    (inline or by path) and let it be minted, or paste a pre-generated one into
    `oauth_apple_client_secret`.
    """

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
