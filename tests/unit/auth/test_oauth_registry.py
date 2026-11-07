"""The OAuth protocol mechanics: endpoints, the authorize URL, the exchange, the profile.

Pins the difference between a discovery provider and one carrying its
endpoints explicitly, that a discovery document is fetched once per URL, and
that every provider's reply is reduced to the same identity — including
GitHub, whose primary address needs a second request. No network is touched.
"""

import json
from typing import Any

import jwt
import pytest

from base.models.definitions.errors import AuthErrors
from base.models.view_models.http import HttpResponse
from base.models.view_models.oauth import OAuthProviderConfig
from base.services.auth.models import TokenResponse
from base.services.auth.oauth import OAuthProviderRegistry
from base.services.configuration.mixins import (
    AppleOAuthSettingsMixin,
    FacebookOAuthSettingsMixin,
    GitHubOAuthSettingsMixin,
    GoogleOAuthSettingsMixin,
    OAuthSettingsMixin,
)
from base.services.http import HttpRequestService
from base.tools.errors import ErrorsException

DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
DISCOVERY_DOCUMENT: dict[str, Any] = {
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
}

MIXINS: tuple[type, ...] = (
    OAuthSettingsMixin,
    GoogleOAuthSettingsMixin,
    GitHubOAuthSettingsMixin,
    FacebookOAuthSettingsMixin,
    AppleOAuthSettingsMixin,
)


class FakeHttp:
    """Answers by URL, and records every call in order."""

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self.routes: dict[str, Any] = routes or {}
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        for prefix, response in self.routes.items():
            if url.startswith(prefix):
                return response
        return HttpResponse.fail(RuntimeError(f"no route for {url}"), 404)

    @property
    def urls(self) -> list[str]:
        return [call["url"] for call in self.calls]


def _registry(
    make_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
    http: FakeHttp,
    providers: str = "google,github,facebook,apple",
    **overrides: Any,
) -> OAuthProviderRegistry:
    values: dict[str, Any] = {
        "oauth_providers": providers,
        "oauth_google_client_id": "google-id",
        "oauth_google_client_secret": "google-secret",
        "oauth_google_redirect_uri": "https://svc/cb/google",
        "oauth_github_client_id": "github-id",
        "oauth_github_client_secret": "github-secret",
        "oauth_github_redirect_uri": "https://svc/cb/github",
        "oauth_facebook_client_id": "facebook-id",
        "oauth_facebook_client_secret": "facebook-secret",
        "oauth_facebook_redirect_uri": "https://svc/cb/facebook",
        "oauth_apple_client_id": "com.example.service",
        "oauth_apple_client_secret": "apple-secret",
        "oauth_apple_redirect_uri": "https://svc/cb/apple",
    }
    values.update(overrides)
    make_settings(*MIXINS, **values)
    monkeypatch.setattr(HttpRequestService, "acall_endpoint", http)
    return OAuthProviderRegistry()


def _id_token(**claims: Any) -> str:
    """An unverified id token; the profile readers never check its signature."""
    return jwt.encode(claims, "irrelevant-for-an-unverified-read", algorithm="HS256")


def test_an_enabled_provider_is_returned_by_name(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    assert registry.get("google").name == "google"
    assert registry.get("GOOGLE").name == "google"


def test_a_provider_that_was_not_enabled_is_refused(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp(), providers="google")
    with pytest.raises(ErrorsException) as caught:
        registry.get("github")
    assert caught.value.error_enum is AuthErrors.OAUTH_PROVIDER_DISABLED


def test_the_enabled_names_are_sorted(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    assert registry.enabled_names() == ["apple", "facebook", "github", "google"]


async def test_a_provider_with_explicit_endpoints_needs_no_discovery(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp()
    registry = _registry(make_settings, monkeypatch, http)

    endpoints = await registry.aresolve_endpoints(registry.get("github"))
    assert endpoints == (
        "https://github.com/login/oauth/authorize",
        "https://github.com/login/oauth/access_token",
        "https://api.github.com/user",
    )
    assert http.calls == []


async def test_a_discovery_provider_reads_its_endpoints_from_the_document(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp({DISCOVERY_URL: HttpResponse.ok(200, DISCOVERY_DOCUMENT)})
    registry = _registry(make_settings, monkeypatch, http)

    authorize, token, userinfo = await registry.aresolve_endpoints(
        registry.get("google")
    )
    assert authorize == DISCOVERY_DOCUMENT["authorization_endpoint"]
    assert token == DISCOVERY_DOCUMENT["token_endpoint"]
    assert userinfo == DISCOVERY_DOCUMENT["userinfo_endpoint"]
    assert http.urls == [DISCOVERY_URL]


async def test_a_discovery_document_is_fetched_once_per_url(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp({DISCOVERY_URL: HttpResponse.ok(200, DISCOVERY_DOCUMENT)})
    registry = _registry(make_settings, monkeypatch, http)

    for _ in range(3):
        await registry.aresolve_endpoints(registry.get("google"))
    assert http.urls == [DISCOVERY_URL]


async def test_a_failed_discovery_request_is_an_exchange_failure(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp({DISCOVERY_URL: HttpResponse.fail(RuntimeError("down"), 503)})
    registry = _registry(make_settings, monkeypatch, http)

    with pytest.raises(ErrorsException) as caught:
        await registry.aresolve_endpoints(registry.get("google"))
    assert caught.value.error_enum is AuthErrors.OAUTH_EXCHANGE_FAILED


async def test_a_discovery_document_missing_an_endpoint_is_an_exchange_failure(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp(
        {DISCOVERY_URL: HttpResponse.ok(200, {"token_endpoint": "https://t"})}
    )
    registry = _registry(make_settings, monkeypatch, http)

    with pytest.raises(ErrorsException) as caught:
        await registry.aresolve_endpoints(registry.get("google"))
    assert caught.value.error_enum is AuthErrors.OAUTH_EXCHANGE_FAILED


async def test_a_provider_with_neither_discovery_nor_endpoints_is_refused(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp()
    registry = _registry(
        make_settings, monkeypatch, http, oauth_google_discovery_url=""
    )

    with pytest.raises(ErrorsException) as caught:
        await registry.aresolve_endpoints(registry.get("google"))
    assert caught.value.error_enum is AuthErrors.OAUTH_PROVIDER_DISABLED


def test_the_authorize_url_carries_the_pkce_challenge_and_state(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from urllib.parse import parse_qs, urlsplit

    registry = _registry(make_settings, monkeypatch, FakeHttp())
    url = registry.build_authorize_url(
        "https://accounts.google.com/o/oauth2/v2/auth",
        registry.get("google"),
        state="the-state",
        code_challenge="the-challenge",
    )
    params = parse_qs(urlsplit(url).query)
    assert params["client_id"] == ["google-id"]
    assert params["redirect_uri"] == ["https://svc/cb/google"]
    assert params["response_type"] == ["code"]
    assert params["scope"] == ["openid email profile"]
    assert params["state"] == ["the-state"]
    assert params["code_challenge"] == ["the-challenge"]
    assert params["code_challenge_method"] == ["S256"]
    assert "response_mode" not in params


def test_apple_asks_for_its_callback_to_be_posted(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    url = registry.build_authorize_url(
        "https://appleid.apple.com/auth/authorize",
        registry.get("apple"),
        state="s",
        code_challenge="c",
    )
    assert "response_mode=form_post" in url


def test_an_authorize_url_that_already_has_a_query_is_appended_to(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    url = registry.build_authorize_url(
        "https://idp.example.com/authorize?tenant=acme",
        registry.get("google"),
        state="s",
        code_challenge="c",
    )
    assert url.startswith("https://idp.example.com/authorize?tenant=acme&")
    assert url.count("?") == 1


def test_a_static_client_secret_is_used_as_is(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    assert registry.resolve_client_secret(registry.get("google")) == "google-secret"


def test_apples_client_secret_is_minted_fresh_for_each_exchange(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch, ec_private_pem: str
) -> None:
    registry = _registry(
        make_settings,
        monkeypatch,
        FakeHttp(),
        oauth_apple_client_secret="",
        oauth_apple_team_id="TEAMID",
        oauth_apple_key_id="KEYID",
        oauth_apple_private_key=ec_private_pem,
    )
    apple = registry.get("apple")
    secret = registry.resolve_client_secret(apple)
    claims = jwt.decode(secret, options={"verify_signature": False})
    assert claims["iss"] == "TEAMID"
    assert claims["sub"] == "com.example.service"


async def test_the_code_exchange_posts_the_authorization_code_grant(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp(
        {
            "https://oauth2.googleapis.com/token": HttpResponse.ok(
                200, {"access_token": "at", "token_type": "Bearer", "expires_in": 3599}
            )
        }
    )
    registry = _registry(make_settings, monkeypatch, http)

    parsed = await registry.aexchange_code(
        registry.get("google"),
        code="the-code",
        code_verifier="the-verifier",
        token_url="https://oauth2.googleapis.com/token",
    )
    assert isinstance(parsed, TokenResponse)
    assert parsed.access_token == "at"
    sent = http.calls[-1]
    assert sent["method"] == "POST"
    assert sent["data"]["grant_type"] == "authorization_code"
    assert sent["data"]["code"] == "the-code"
    assert sent["data"]["code_verifier"] == "the-verifier"
    assert sent["data"]["client_id"] == "google-id"
    assert sent["data"]["client_secret"] == "google-secret"
    assert sent["data"]["redirect_uri"] == "https://svc/cb/google"
    assert sent["headers"]["Accept"] == "application/json"


async def test_a_failed_code_exchange_is_reported_as_one(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp({"https://t": HttpResponse.fail(RuntimeError("bad code"), 400)})
    registry = _registry(make_settings, monkeypatch, http)

    with pytest.raises(ErrorsException) as caught:
        await registry.aexchange_code(
            registry.get("google"),
            code="c",
            code_verifier="v",
            token_url="https://t",
        )
    assert caught.value.error_enum is AuthErrors.OAUTH_EXCHANGE_FAILED


async def test_a_token_reply_without_an_access_token_is_an_exchange_failure(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp({"https://t": HttpResponse.ok(200, {"error": "invalid_grant"})})
    registry = _registry(make_settings, monkeypatch, http)

    with pytest.raises(ErrorsException) as caught:
        await registry.aexchange_code(
            registry.get("google"),
            code="c",
            code_verifier="v",
            token_url="https://t",
        )
    assert caught.value.error_enum is AuthErrors.OAUTH_EXCHANGE_FAILED


async def test_an_oidc_profile_comes_from_the_id_token_when_one_is_present(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp()
    registry = _registry(make_settings, monkeypatch, http)
    payload = TokenResponse(
        access_token="at",
        id_token=_id_token(
            sub="google-subject",
            email="person@example.com",
            email_verified=True,
            given_name="Ada",
            family_name="Lovelace",
        ),
    )

    profile = await registry.afetch_profile(
        registry.get("google"),
        token_payload=payload,
        userinfo_url="https://userinfo",
    )
    assert profile.subject == "google-subject"
    assert profile.email == "person@example.com"
    assert profile.email_verified is True
    assert profile.first_name == "Ada"
    assert profile.last_name == "Lovelace"
    assert http.calls == []


async def test_an_oidc_profile_falls_back_to_the_userinfo_endpoint(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp(
        {
            "https://userinfo": HttpResponse.ok(
                200,
                {
                    "sub": "google-subject",
                    "email": "person@example.com",
                    "given_name": "Ada",
                },
            )
        }
    )
    registry = _registry(make_settings, monkeypatch, http)

    profile = await registry.afetch_profile(
        registry.get("google"),
        token_payload=TokenResponse(access_token="at"),
        userinfo_url="https://userinfo",
    )
    assert profile.subject == "google-subject"
    assert profile.first_name == "Ada"
    assert http.calls[-1]["headers"]["Authorization"] == "Bearer at"


async def test_an_oidc_id_token_without_an_email_falls_back_to_userinfo(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp(
        {
            "https://userinfo": HttpResponse.ok(
                200, {"sub": "google-subject", "email": "person@example.com"}
            )
        }
    )
    registry = _registry(make_settings, monkeypatch, http)

    profile = await registry.afetch_profile(
        registry.get("google"),
        token_payload=TokenResponse(
            access_token="at", id_token=_id_token(sub="google-subject")
        ),
        userinfo_url="https://userinfo",
    )
    assert profile.email == "person@example.com"
    assert http.urls == ["https://userinfo"]


async def test_an_oidc_profile_with_no_userinfo_url_is_incomplete(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    with pytest.raises(ErrorsException) as caught:
        await registry.afetch_profile(
            registry.get("google"),
            token_payload=TokenResponse(access_token="at"),
            userinfo_url="",
        )
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE


async def test_an_apple_profile_comes_from_its_id_token(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    payload = TokenResponse(
        access_token="at",
        id_token=_id_token(
            sub="apple-subject", email="person@privaterelay.appleid.com"
        ),
    )

    profile = await registry.afetch_profile(
        registry.get("apple"), token_payload=payload, userinfo_url=""
    )
    assert profile.subject == "apple-subject"
    assert profile.email == "person@privaterelay.appleid.com"


async def test_an_apple_name_arrives_only_in_the_first_callback(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apple posts the name once, beside the code, and never again."""
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    payload = TokenResponse(
        access_token="at",
        id_token=_id_token(sub="apple-subject", email="person@example.com"),
    )

    profile = await registry.afetch_profile(
        registry.get("apple"),
        token_payload=payload,
        userinfo_url="",
        apple_user_json=json.dumps(
            {"name": {"firstName": "Ada", "lastName": "Lovelace"}}
        ),
    )
    assert (profile.first_name, profile.last_name) == ("Ada", "Lovelace")


async def test_an_unparseable_apple_user_blob_is_ignored(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    payload = TokenResponse(
        access_token="at",
        id_token=_id_token(sub="apple-subject", email="person@example.com"),
    )

    profile = await registry.afetch_profile(
        registry.get("apple"),
        token_payload=payload,
        userinfo_url="",
        apple_user_json="{not json",
    )
    assert profile.first_name is None


async def test_an_apple_reply_without_an_id_token_is_incomplete(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    with pytest.raises(ErrorsException) as caught:
        await registry.afetch_profile(
            registry.get("apple"),
            token_payload=TokenResponse(access_token="at"),
            userinfo_url="",
        )
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE


async def test_a_facebook_profile_passes_the_token_in_the_query(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp(
        {
            "https://graph.facebook.com/me": HttpResponse.ok(
                200,
                {
                    "id": "facebook-subject",
                    "email": "person@example.com",
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                },
            )
        }
    )
    registry = _registry(make_settings, monkeypatch, http)

    profile = await registry.afetch_profile(
        registry.get("facebook"),
        token_payload=TokenResponse(access_token="at"),
        userinfo_url="",
    )
    assert profile.subject == "facebook-subject"
    assert profile.first_name == "Ada"
    assert "access_token=at" in http.urls[-1]


async def test_a_github_profile_splits_the_single_name_field(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp(
        {
            "https://api.github.com/user": HttpResponse.ok(
                200,
                {
                    "id": 4242,
                    "email": "person@example.com",
                    "name": "Ada Lovelace",
                },
            )
        }
    )
    registry = _registry(make_settings, monkeypatch, http)

    profile = await registry.afetch_profile(
        registry.get("github"),
        token_payload=TokenResponse(access_token="at"),
        userinfo_url="",
    )
    assert profile.subject == "4242"
    assert (profile.first_name, profile.last_name) == ("Ada", "Lovelace")
    assert http.calls[0]["headers"]["Accept"] == "application/vnd.github+json"


async def test_a_github_private_address_is_read_from_the_emails_endpoint(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub omits `email` from the profile when a user hides it."""
    http = FakeHttp(
        {
            "https://api.github.com/user/emails": json.dumps(
                [
                    {"email": "other@example.com", "primary": False, "verified": True},
                    {"email": "person@example.com", "primary": True, "verified": True},
                ]
            ).encode(),
            "https://api.github.com/user": HttpResponse.ok(
                200, {"id": 4242, "name": "Ada"}
            ),
        }
    )
    registry = _registry(make_settings, monkeypatch, http)

    profile = await registry.afetch_profile(
        registry.get("github"),
        token_payload=TokenResponse(access_token="at"),
        userinfo_url="",
    )
    assert profile.email == "person@example.com"
    assert profile.last_name is None
    assert http.calls[-1]["raw"] is True


async def test_a_github_user_with_no_usable_address_is_incomplete(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp(
        {
            "https://api.github.com/user/emails": json.dumps(
                [{"email": "x@example.com", "primary": True, "verified": False}]
            ).encode(),
            "https://api.github.com/user": HttpResponse.ok(200, {"id": 4242}),
        }
    )
    registry = _registry(make_settings, monkeypatch, http)

    with pytest.raises(ErrorsException) as caught:
        await registry.afetch_profile(
            registry.get("github"),
            token_payload=TokenResponse(access_token="at"),
            userinfo_url="",
        )
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE


async def test_an_unknown_profile_style_is_refused(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(make_settings, monkeypatch, FakeHttp())
    unknown = OAuthProviderConfig(
        name="mystery",
        client_id="id",
        client_secret="secret",
        redirect_uri="https://svc/cb",
        scopes="",
        profile_style="something-new",
    )

    with pytest.raises(ErrorsException) as caught:
        await registry.afetch_profile(
            unknown,
            token_payload=TokenResponse(access_token="at"),
            userinfo_url="https://userinfo",
        )
    assert caught.value.error_enum is AuthErrors.OAUTH_PROVIDER_DISABLED
