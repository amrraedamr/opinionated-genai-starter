from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsGitHubOAuthConfig(Protocol):
    """Satisfied by `GitHubOAuthSettingsMixin`."""

    oauth_github_client_id: str
    oauth_github_client_secret: str
    oauth_github_authorize_url: str
    oauth_github_token_url: str
    oauth_github_userinfo_url: str
    oauth_github_scopes: str
    oauth_github_redirect_uri: str
