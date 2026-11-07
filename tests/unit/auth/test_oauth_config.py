"""Turning `OAUTH_PROVIDERS` into the providers a service can actually offer.

Only the named providers are read, so a service composes one settings mixin
per provider it offers and declares no credentials for the rest. Pins that an
unnamed provider is never even constructed — the behaviour that replaced
building all five and filtering afterwards.
"""

from typing import Any

import pytest

from base.models.definitions.errors import AuthErrors
from base.models.view_models.oauth import OAuthProviderConfig
from base.services.auth.oauth import config as config_module
from base.services.auth.oauth.config import load_oauth_provider_configs
from base.services.configuration.mixins import (
    AppleOAuthSettingsMixin,
    FacebookOAuthSettingsMixin,
    GitHubOAuthSettingsMixin,
    GoogleOAuthSettingsMixin,
    MicrosoftOAuthSettingsMixin,
    OAuthSettingsMixin,
)
from base.tools.errors import ErrorsException, MissingSettingsMixinError

ALL_PROVIDER_MIXINS: tuple[type, ...] = (
    OAuthSettingsMixin,
    GoogleOAuthSettingsMixin,
    MicrosoftOAuthSettingsMixin,
    GitHubOAuthSettingsMixin,
    FacebookOAuthSettingsMixin,
    AppleOAuthSettingsMixin,
)

CREDENTIALS: dict[str, Any] = {
    "oauth_google_client_id": "google-id",
    "oauth_google_client_secret": "google-secret",
    "oauth_google_redirect_uri": "https://svc.example.com/cb/google",
    "oauth_microsoft_client_id": "microsoft-id",
    "oauth_microsoft_client_secret": "microsoft-secret",
    "oauth_microsoft_redirect_uri": "https://svc.example.com/cb/microsoft",
    "oauth_github_client_id": "github-id",
    "oauth_github_client_secret": "github-secret",
    "oauth_github_redirect_uri": "https://svc.example.com/cb/github",
    "oauth_facebook_client_id": "facebook-id",
    "oauth_facebook_client_secret": "facebook-secret",
    "oauth_facebook_redirect_uri": "https://svc.example.com/cb/facebook",
    "oauth_apple_client_id": "com.example.service",
    "oauth_apple_client_secret": "apple-secret",
    "oauth_apple_redirect_uri": "https://svc.example.com/cb/apple",
}


def _configured(make_settings: Any, providers: str, **overrides: Any) -> Any:
    values = {**CREDENTIALS, **overrides}
    return make_settings(*ALL_PROVIDER_MIXINS, oauth_providers=providers, **values)


def test_only_the_named_providers_are_offered(make_settings: Any) -> None:
    _configured(make_settings, "google,github")
    assert sorted(load_oauth_provider_configs()) == ["github", "google"]


def test_provider_names_are_matched_case_insensitively(make_settings: Any) -> None:
    _configured(make_settings, "GOOGLE,MicroSoft")
    assert sorted(load_oauth_provider_configs()) == ["google", "microsoft"]


def test_surrounding_whitespace_in_the_list_is_dropped(make_settings: Any) -> None:
    _configured(make_settings, "  google ,   github  ,")
    assert sorted(load_oauth_provider_configs()) == ["github", "google"]


def test_a_duplicated_name_yields_one_provider(make_settings: Any) -> None:
    _configured(make_settings, "google,google,GOOGLE")
    assert list(load_oauth_provider_configs()) == ["google"]


def test_an_empty_list_offers_nothing(make_settings: Any) -> None:
    _configured(make_settings, "")
    assert load_oauth_provider_configs() == {}


def test_a_list_of_only_separators_offers_nothing(make_settings: Any) -> None:
    _configured(make_settings, " , , ")
    assert load_oauth_provider_configs() == {}


def test_every_known_provider_can_be_enabled_at_once(make_settings: Any) -> None:
    _configured(make_settings, "google,microsoft,github,facebook,apple")
    assert sorted(load_oauth_provider_configs()) == [
        "apple",
        "facebook",
        "github",
        "google",
        "microsoft",
    ]


def test_an_unnamed_provider_is_never_constructed(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the fix: building all five then filtering read settings it should not."""
    built: list[str] = []
    for name, (protocol, mixin, build) in dict(config_module._PROVIDERS).items():

        def _recording(settings: Any, _name: str = name, _build: Any = build) -> Any:
            built.append(_name)
            return _build(settings)

        monkeypatch.setitem(
            config_module._PROVIDERS, name, (protocol, mixin, _recording)
        )

    _configured(make_settings, "google")
    load_oauth_provider_configs()
    assert built == ["google"]


def test_enabling_a_provider_whose_mixin_is_absent_names_the_mixin(
    make_settings: Any,
) -> None:
    make_settings(OAuthSettingsMixin, oauth_providers="apple")
    with pytest.raises(MissingSettingsMixinError) as caught:
        load_oauth_provider_configs()
    assert caught.value.selected == "apple"
    assert caught.value.mixin == "AppleOAuthSettingsMixin"
    assert caught.value.setting == "OAUTH_PROVIDERS"
    assert "AppleOAuthSettingsMixin" in str(caught.value)


@pytest.mark.parametrize(
    ("provider", "mixin_name"),
    [
        ("google", "GoogleOAuthSettingsMixin"),
        ("microsoft", "MicrosoftOAuthSettingsMixin"),
        ("github", "GitHubOAuthSettingsMixin"),
        ("facebook", "FacebookOAuthSettingsMixin"),
        ("apple", "AppleOAuthSettingsMixin"),
    ],
)
def test_each_provider_names_its_own_mixin_when_it_is_absent(
    make_settings: Any, provider: str, mixin_name: str
) -> None:
    make_settings(OAuthSettingsMixin, oauth_providers=provider)
    with pytest.raises(MissingSettingsMixinError) as caught:
        load_oauth_provider_configs()
    assert caught.value.mixin == mixin_name


def test_a_missing_mixin_is_reported_before_any_provider_is_built(
    make_settings: Any,
) -> None:
    make_settings(
        OAuthSettingsMixin,
        GoogleOAuthSettingsMixin,
        oauth_providers="apple,google",
        oauth_google_client_id="google-id",
        oauth_google_client_secret="google-secret",
        oauth_google_redirect_uri="https://svc.example.com/cb/google",
    )
    with pytest.raises(MissingSettingsMixinError):
        load_oauth_provider_configs()


def test_an_unknown_provider_name_is_refused(make_settings: Any) -> None:
    _configured(make_settings, "myspace")
    with pytest.raises(ErrorsException) as caught:
        load_oauth_provider_configs()
    assert caught.value.error_enum is AuthErrors.OAUTH_PROVIDER_DISABLED


def test_an_unknown_name_beside_a_known_one_is_still_refused(
    make_settings: Any,
) -> None:
    _configured(make_settings, "google,myspace")
    with pytest.raises(ErrorsException):
        load_oauth_provider_configs()


@pytest.mark.parametrize(
    "missing",
    ["oauth_google_client_id", "oauth_google_client_secret"],
)
def test_a_provider_missing_a_credential_is_dropped(
    make_settings: Any, missing: str
) -> None:
    _configured(make_settings, "google", **{missing: ""})
    assert load_oauth_provider_configs() == {}


def test_a_provider_without_a_redirect_uri_is_dropped(make_settings: Any) -> None:
    _configured(make_settings, "google", oauth_google_redirect_uri="")
    assert load_oauth_provider_configs() == {}


def test_a_dropped_provider_does_not_hide_a_usable_one(make_settings: Any) -> None:
    _configured(make_settings, "google,github", oauth_google_client_secret="")
    assert list(load_oauth_provider_configs()) == ["github"]


def test_apple_is_usable_from_signing_material_without_a_static_secret(
    make_settings: Any, ec_private_pem: str
) -> None:
    _configured(
        make_settings,
        "apple",
        oauth_apple_client_secret="",
        oauth_apple_team_id="TEAMID",
        oauth_apple_key_id="KEYID",
        oauth_apple_private_key=ec_private_pem,
    )
    apple = load_oauth_provider_configs()["apple"]
    assert apple.client_secret
    assert apple.apple_private_key == ec_private_pem.strip()


def test_apple_without_a_secret_or_signing_material_is_dropped(
    make_settings: Any,
) -> None:
    _configured(make_settings, "apple", oauth_apple_client_secret="")
    assert load_oauth_provider_configs() == {}


def test_apple_with_partial_signing_material_is_dropped(
    make_settings: Any, ec_private_pem: str
) -> None:
    _configured(
        make_settings,
        "apple",
        oauth_apple_client_secret="",
        oauth_apple_team_id="TEAMID",
        oauth_apple_private_key=ec_private_pem,
    )
    assert load_oauth_provider_configs() == {}


def test_the_registry_knows_exactly_five_providers() -> None:
    assert sorted(config_module._PROVIDERS) == [
        "apple",
        "facebook",
        "github",
        "google",
        "microsoft",
    ]


def test_every_registry_entry_pairs_a_protocol_a_mixin_name_and_a_builder() -> None:
    for name, (protocol, mixin, build) in config_module._PROVIDERS.items():
        assert isinstance(protocol, type)
        assert mixin.endswith("SettingsMixin")
        assert callable(build)
        assert name in mixin.lower().replace("oauthsettingsmixin", "")


def test_a_loaded_config_is_an_oauth_provider_config(make_settings: Any) -> None:
    _configured(make_settings, "google")
    assert isinstance(load_oauth_provider_configs()["google"], OAuthProviderConfig)
