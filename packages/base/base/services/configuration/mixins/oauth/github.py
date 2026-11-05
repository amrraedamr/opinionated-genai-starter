from pydantic import BaseModel


class GitHubOAuthSettingsMixin(BaseModel):
    oauth_github_client_id: str = ""
    oauth_github_client_secret: str = ""
    oauth_github_authorize_url: str = "https://github.com/login/oauth/authorize"
    oauth_github_token_url: str = "https://github.com/login/oauth/access_token"
    oauth_github_userinfo_url: str = "https://api.github.com/user"
    oauth_github_scopes: str = "read:user user:email"
    oauth_github_redirect_uri: str = ""
