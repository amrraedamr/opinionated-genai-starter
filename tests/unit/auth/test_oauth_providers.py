"""What each provider's settings become, provider by provider.

Google, Microsoft and Apple are discovered from an OIDC document; GitHub and
Facebook have no discovery and carry their three endpoints explicitly. The
split is what decides whether a missing URL is a configuration error or
something the registry fetches later, so it is pinned per provider.
"""

from typing import Any

import pytest

from base.models.view_models.oauth import OAuthProviderConfig
from base.services.auth.oauth.config import (
    _apple,
    _facebook,
    _github,
    _google,
    _microsoft,
    _usable,
)
from base.services.configuration.mixins import (
    AppleOAuthSettingsMixin,
    FacebookOAuthSettingsMixin,
    GitHubOAuthSettingsMixin,
    GoogleOAuthSettingsMixin,
    MicrosoftOAuthSettingsMixin,
)
from tests.conftest import settings_class

DISCOVERY_PROVIDERS: tuple[str, ...] = ("google", "microsoft", "apple")
EXPLICIT_PROVIDERS: tuple[str, ...] = ("github", "facebook")


def _build(mixin: type, builder: Any, **values: Any) -> OAuthProviderConfig:
    settings = settings_class(mixin)(service_name="svc", **values)
    return builder(settings)


def test_google_is_an_oidc_provider_discovered_from_google() -> None:
    cfg = _build(
        GoogleOAuthSettingsMixin,
        _google,
        oauth_google_client_id="gid",
        oauth_google_client_secret="gsecret",
        oauth_google_redirect_uri="https://svc/cb",
    )
    assert cfg.name == "google"
    assert cfg.client_id == "gid"
    assert cfg.client_secret == "gsecret"
    assert cfg.redirect_uri == "https://svc/cb"
    assert cfg.scopes == "openid email profile"
    assert cfg.profile_style == "oidc"
    assert cfg.discovery_url.startswith("https://accounts.google.com/")
    assert cfg.authorize_url == ""
    assert cfg.token_url == ""
    assert cfg.response_mode == ""


def test_microsoft_is_an_oidc_provider_discovered_from_entra() -> None:
    cfg = _build(
        MicrosoftOAuthSettingsMixin,
        _microsoft,
        oauth_microsoft_client_id="mid",
        oauth_microsoft_client_secret="msecret",
        oauth_microsoft_redirect_uri="https://svc/cb",
    )
    assert cfg.name == "microsoft"
    assert cfg.profile_style == "oidc"
    assert cfg.discovery_url.startswith("https://login.microsoftonline.com/")
    assert cfg.scopes == "openid email profile"
    assert cfg.authorize_url == ""


def test_apple_is_an_oidc_provider_that_posts_its_callback(
    ec_private_pem: str,
) -> None:
    cfg = _build(
        AppleOAuthSettingsMixin,
        _apple,
        oauth_apple_client_id="com.example.service",
        oauth_apple_team_id="TEAMID",
        oauth_apple_key_id="KEYID",
        oauth_apple_private_key=ec_private_pem,
        oauth_apple_redirect_uri="https://svc/cb",
    )
    assert cfg.name == "apple"
    assert cfg.profile_style == "apple"
    assert cfg.response_mode == "form_post"
    assert cfg.discovery_url.startswith("https://appleid.apple.com/")
    assert cfg.scopes == "name email"
    assert cfg.apple_team_id == "TEAMID"
    assert cfg.apple_key_id == "KEYID"
    assert cfg.apple_private_key == ec_private_pem.strip()
    assert cfg.client_secret, "a secret is minted from the signing material"


def test_apple_prefers_a_pre_generated_client_secret(ec_private_pem: str) -> None:
    cfg = _build(
        AppleOAuthSettingsMixin,
        _apple,
        oauth_apple_client_id="com.example.service",
        oauth_apple_client_secret="pasted-in-by-hand",
        oauth_apple_team_id="TEAMID",
        oauth_apple_key_id="KEYID",
        oauth_apple_private_key=ec_private_pem,
        oauth_apple_redirect_uri="https://svc/cb",
    )
    assert cfg.client_secret == "pasted-in-by-hand"


def test_github_carries_its_three_endpoints_explicitly() -> None:
    cfg = _build(
        GitHubOAuthSettingsMixin,
        _github,
        oauth_github_client_id="ghid",
        oauth_github_client_secret="ghsecret",
        oauth_github_redirect_uri="https://svc/cb",
    )
    assert cfg.name == "github"
    assert cfg.profile_style == "github"
    assert cfg.discovery_url == ""
    assert cfg.authorize_url == "https://github.com/login/oauth/authorize"
    assert cfg.token_url == "https://github.com/login/oauth/access_token"
    assert cfg.userinfo_url == "https://api.github.com/user"
    assert cfg.scopes == "read:user user:email"


def test_facebook_carries_its_three_endpoints_explicitly() -> None:
    cfg = _build(
        FacebookOAuthSettingsMixin,
        _facebook,
        oauth_facebook_client_id="fbid",
        oauth_facebook_client_secret="fbsecret",
        oauth_facebook_redirect_uri="https://svc/cb",
    )
    assert cfg.name == "facebook"
    assert cfg.profile_style == "facebook"
    assert cfg.discovery_url == ""
    assert cfg.authorize_url == "https://www.facebook.com/v21.0/dialog/oauth"
    assert cfg.token_url.startswith("https://graph.facebook.com/")
    assert "fields=id,email,first_name,last_name" in cfg.userinfo_url
    assert cfg.scopes == "email public_profile"


def test_a_deployment_can_point_any_provider_at_its_own_endpoints() -> None:
    """A GitHub Enterprise or a Graph API version an operator pins."""
    cfg = _build(
        GitHubOAuthSettingsMixin,
        _github,
        oauth_github_client_id="ghid",
        oauth_github_client_secret="ghsecret",
        oauth_github_redirect_uri="https://svc/cb",
        oauth_github_authorize_url="https://ghe.example.com/login/oauth/authorize",
        oauth_github_token_url="https://ghe.example.com/login/oauth/access_token",
        oauth_github_userinfo_url="https://ghe.example.com/api/v3/user",
        oauth_github_scopes="read:user",
    )
    assert cfg.authorize_url == "https://ghe.example.com/login/oauth/authorize"
    assert cfg.token_url == "https://ghe.example.com/login/oauth/access_token"
    assert cfg.userinfo_url == "https://ghe.example.com/api/v3/user"
    assert cfg.scopes == "read:user"


@pytest.mark.parametrize("name", DISCOVERY_PROVIDERS)
def test_a_discovery_provider_declares_no_authorize_or_token_url(
    name: str, ec_private_pem: str
) -> None:
    cfg = _any_provider(name, ec_private_pem)
    assert cfg.discovery_url
    assert cfg.authorize_url == ""
    assert cfg.token_url == ""


@pytest.mark.parametrize("name", EXPLICIT_PROVIDERS)
def test_an_explicit_provider_declares_no_discovery_url(
    name: str, ec_private_pem: str
) -> None:
    cfg = _any_provider(name, ec_private_pem)
    assert cfg.discovery_url == ""
    assert cfg.authorize_url
    assert cfg.token_url
    assert cfg.userinfo_url


@pytest.mark.parametrize("name", DISCOVERY_PROVIDERS + EXPLICIT_PROVIDERS)
def test_every_provider_needs_a_client_id_and_a_redirect_uri(
    name: str, ec_private_pem: str
) -> None:
    assert _usable(_any_provider(name, ec_private_pem)) is True
    assert _usable(_replace(_any_provider(name, ec_private_pem), client_id="")) is False
    assert (
        _usable(_replace(_any_provider(name, ec_private_pem), redirect_uri="")) is False
    )


@pytest.mark.parametrize("name", DISCOVERY_PROVIDERS + EXPLICIT_PROVIDERS)
def test_every_provider_except_apple_needs_a_static_client_secret(
    name: str, ec_private_pem: str
) -> None:
    without_secret = _replace(_any_provider(name, ec_private_pem), client_secret="")
    assert _usable(without_secret) is (name == "apple")


def test_apple_is_usable_from_signing_material_alone(ec_private_pem: str) -> None:
    cfg = _replace(_any_provider("apple", ec_private_pem), client_secret="")
    assert _usable(cfg) is True


@pytest.mark.parametrize("field", ["apple_team_id", "apple_key_id"])
def test_apple_needs_all_of_its_signing_material(
    field: str, ec_private_pem: str
) -> None:
    cfg = _replace(
        _any_provider("apple", ec_private_pem), client_secret="", **{field: ""}
    )
    assert _usable(cfg) is False


def test_a_provider_config_is_frozen() -> None:
    cfg = _build(
        GoogleOAuthSettingsMixin,
        _google,
        oauth_google_client_id="gid",
        oauth_google_client_secret="gsecret",
        oauth_google_redirect_uri="https://svc/cb",
    )
    with pytest.raises(AttributeError):
        cfg.client_id = "someone-else"  # type: ignore[misc]


def _replace(cfg: OAuthProviderConfig, **changes: Any) -> OAuthProviderConfig:
    import dataclasses

    return dataclasses.replace(cfg, **changes)


def _any_provider(name: str, ec_private_pem: str) -> OAuthProviderConfig:
    """A fully configured provider by name, for the cross-provider checks."""
    builders: dict[str, tuple[type, Any, dict[str, Any]]] = {
        "google": (
            GoogleOAuthSettingsMixin,
            _google,
            {
                "oauth_google_client_id": "gid",
                "oauth_google_client_secret": "gsecret",
                "oauth_google_redirect_uri": "https://svc/cb",
            },
        ),
        "microsoft": (
            MicrosoftOAuthSettingsMixin,
            _microsoft,
            {
                "oauth_microsoft_client_id": "mid",
                "oauth_microsoft_client_secret": "msecret",
                "oauth_microsoft_redirect_uri": "https://svc/cb",
            },
        ),
        "github": (
            GitHubOAuthSettingsMixin,
            _github,
            {
                "oauth_github_client_id": "ghid",
                "oauth_github_client_secret": "ghsecret",
                "oauth_github_redirect_uri": "https://svc/cb",
            },
        ),
        "facebook": (
            FacebookOAuthSettingsMixin,
            _facebook,
            {
                "oauth_facebook_client_id": "fbid",
                "oauth_facebook_client_secret": "fbsecret",
                "oauth_facebook_redirect_uri": "https://svc/cb",
            },
        ),
        "apple": (
            AppleOAuthSettingsMixin,
            _apple,
            {
                "oauth_apple_client_id": "com.example.service",
                "oauth_apple_team_id": "TEAMID",
                "oauth_apple_key_id": "KEYID",
                "oauth_apple_private_key": ec_private_pem,
                "oauth_apple_redirect_uri": "https://svc/cb",
            },
        ),
    }
    mixin, builder, values = builders[name]
    return _build(mixin, builder, **values)
