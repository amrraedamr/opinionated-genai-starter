"""The authorization-code flow, from the redirect out to the code coming back.

The registry under test is the real one; only its HTTP client is replaced, so
a provider returning junk is exercised rather than described. Everything that
leaves this service goes through the front channel, so what the authorize URL
carries — and what it must not — is asserted off the URL itself.
"""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from base.models.definitions.errors import AuthErrors
from base.services.auth import TokenVerifier
from base.tools.errors import ErrorsException
from base.tools.security import b64url
from identity.models.definitions.auth import SignedPayloadType
from identity.services.auth.oauth import OAuthService

from .conftest import FakeHttp, enable_apple, github_emails, id_token


@pytest.fixture
def service(identity_settings: Any) -> OAuthService:
    return OAuthService()


def query_of(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


async def start(service: OAuthService, provider: str = "google") -> tuple[str, str]:
    url, nonce = await service.abuild_authorize_redirect(provider)
    return query_of(url)["state"], nonce


# Starting a flow


async def test_the_authorize_url_is_the_discovered_endpoint(
    service: OAuthService, google_http: FakeHttp
) -> None:
    url, _ = await service.abuild_authorize_redirect("google")
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")


async def test_the_authorize_url_carries_everything_the_provider_needs(
    service: OAuthService, google_http: FakeHttp
) -> None:
    url, _ = await service.abuild_authorize_redirect("google")
    params = query_of(url)
    assert params["client_id"] == "google-client"
    assert params["response_type"] == "code"
    assert params["scope"] == "openid email profile"
    assert params["redirect_uri"].endswith("/auth/oauth/google/callback")
    assert params["state"]


async def test_the_authorize_url_commits_to_a_pkce_challenge(
    service: OAuthService, google_http: FakeHttp
) -> None:
    url, _ = await service.abuild_authorize_redirect("google")
    params = query_of(url)
    assert params["code_challenge_method"] == "S256"
    assert params["code_challenge"]


async def test_the_verifier_never_reaches_the_front_channel(
    service: OAuthService, google_http: FakeHttp
) -> None:
    """A verifier in the redirect would make PKCE decorative."""
    url, nonce = await service.abuild_authorize_redirect("google")
    state = query_of(url)["state"]
    verifier = await service._flows.aconsume("google", state, nonce)
    assert verifier not in url
    assert verifier not in nonce


async def test_the_challenge_is_the_sha256_of_the_verifier(
    service: OAuthService, google_http: FakeHttp
) -> None:
    url, nonce = await service.abuild_authorize_redirect("google")
    params = query_of(url)
    verifier = await service._flows.aconsume("google", params["state"], nonce)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    assert params["code_challenge"] == b64url(digest)


async def test_two_flows_never_share_a_state_or_a_nonce(
    service: OAuthService, google_http: FakeHttp
) -> None:
    first_state, first_nonce = await start(service)
    second_state, second_nonce = await start(service)
    assert first_state != second_state
    assert first_nonce != second_nonce


async def test_a_provider_this_service_does_not_offer_is_refused(
    service: OAuthService, google_http: FakeHttp
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.abuild_authorize_redirect("facebook")
    assert caught.value.error_enum is AuthErrors.OAUTH_PROVIDER_DISABLED


async def test_an_unknown_provider_name_is_refused(
    service: OAuthService, google_http: FakeHttp
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.abuild_authorize_redirect("not-a-provider")
    assert caught.value.error_enum is AuthErrors.OAUTH_PROVIDER_DISABLED


async def test_a_provider_name_is_matched_without_regard_to_case(
    service: OAuthService, google_http: FakeHttp
) -> None:
    url, _ = await service.abuild_authorize_redirect("GOOGLE")
    assert url.startswith("https://accounts.google.com/")


async def test_a_discovery_document_that_cannot_be_fetched_stops_the_flow(
    service: OAuthService, http: FakeHttp
) -> None:
    http.fail_route("openid-configuration")
    with pytest.raises(ErrorsException) as caught:
        await service.abuild_authorize_redirect("google")
    assert caught.value.error_enum is AuthErrors.OAUTH_EXCHANGE_FAILED


async def test_a_discovery_document_missing_its_endpoints_stops_the_flow(
    service: OAuthService, http: FakeHttp
) -> None:
    http.json_route("openid-configuration", {"issuer": "https://accounts.google.com"})
    with pytest.raises(ErrorsException) as caught:
        await service.abuild_authorize_redirect("google")
    assert caught.value.error_enum is AuthErrors.OAUTH_EXCHANGE_FAILED


async def test_discovery_is_fetched_once_and_then_remembered(
    service: OAuthService, google_http: FakeHttp
) -> None:
    await service.abuild_authorize_redirect("google")
    await service.abuild_authorize_redirect("google")
    assert google_http.called("openid-configuration") == 1


async def test_a_provider_carrying_its_own_endpoints_is_never_discovered(
    service: OAuthService, http: FakeHttp
) -> None:
    url, _ = await service.abuild_authorize_redirect("github")
    assert url.startswith("https://github.com/login/oauth/authorize?")
    assert http.called("openid-configuration") == 0


# Completing a flow


async def test_a_callback_from_the_same_browser_yields_an_exchange_code(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    code = await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    payload = TokenVerifier().decode_payload(code)
    assert payload["type"] == SignedPayloadType.OAUTH_CODE
    assert payload["user_id"]


async def test_the_token_request_replays_the_verifier_behind_the_challenge(
    service: OAuthService, google_http: FakeHttp
) -> None:
    """The back channel is where the verifier belongs, and the only place."""
    url, nonce = await service.abuild_authorize_redirect("google")
    params = query_of(url)
    await service.ahandle_callback(
        "google", code="the-code", state=params["state"], nonce=nonce
    )
    sent = google_http.sent_to("oauth2.googleapis.com/token")["data"]
    digest = hashlib.sha256(sent["code_verifier"].encode("ascii")).digest()
    assert b64url(digest) == params["code_challenge"]


async def test_the_token_request_is_a_posted_authorization_code_grant(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    await service.ahandle_callback("google", code="the-code", state=state, nonce=nonce)
    call = google_http.sent_to("oauth2.googleapis.com/token")
    assert call["method"] == "POST"
    assert call["data"]["grant_type"] == "authorization_code"
    assert call["data"]["code"] == "the-code"
    assert call["data"]["client_id"] == "google-client"
    assert call["data"]["client_secret"] == "google-secret"


async def test_a_callback_with_no_cookie_is_refused(
    service: OAuthService, google_http: FakeHttp
) -> None:
    """The attack: a state this server really minted, replayed elsewhere."""
    state, _ = await start(service)
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("google", code="c", state=state, nonce=None)
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_a_callback_with_another_browsers_cookie_is_refused(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, _ = await start(service)
    _, other_nonce = await start(service)
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback(
            "google", code="c", state=state, nonce=other_nonce
        )
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_a_state_is_spent_on_its_first_callback(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_a_state_minted_for_one_provider_cannot_finish_at_another(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service, "google")
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("github", code="c", state=state, nonce=nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_a_state_this_server_never_minted_is_refused(
    service: OAuthService, google_http: FakeHttp
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback(
            "google", code="c", state="invented", nonce="invented"
        )
    assert caught.value.error_enum is AuthErrors.OAUTH_STATE_INVALID


async def test_a_callback_for_a_provider_this_service_does_not_offer_is_refused(
    service: OAuthService, google_http: FakeHttp
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("apple", code="c", state="s", nonce="n")
    assert caught.value.error_enum is AuthErrors.OAUTH_PROVIDER_DISABLED


# What the provider sends back


async def test_a_token_endpoint_that_fails_stops_the_login(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    google_http.fail_route("oauth2.googleapis.com/token")
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_EXCHANGE_FAILED


async def test_a_token_endpoint_returning_junk_stops_the_login(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    google_http.json_route("oauth2.googleapis.com/token", {"nothing": "useful"})
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_EXCHANGE_FAILED


async def test_an_id_token_with_no_subject_falls_back_to_userinfo(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    google_http.json_route(
        "oauth2.googleapis.com/token",
        {"access_token": "at", "id_token": id_token(sub="")},
    )
    google_http.json_route(
        "v1/userinfo",
        {"sub": "from-userinfo", "email": "person@example.com", "email_verified": True},
    )
    await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    assert google_http.called("v1/userinfo") == 1


async def test_userinfo_is_fetched_with_the_access_token(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    google_http.json_route("oauth2.googleapis.com/token", {"access_token": "at-123"})
    google_http.json_route(
        "v1/userinfo",
        {"sub": "s", "email": "person@example.com", "email_verified": True},
    )
    await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    headers = google_http.sent_to("v1/userinfo")["headers"]
    assert headers["Authorization"] == "Bearer at-123"


async def test_a_provider_that_names_nobody_stops_the_login(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    google_http.json_route("oauth2.googleapis.com/token", {"access_token": "at"})
    google_http.json_route("v1/userinfo", {"sub": "", "email": ""})
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE


async def test_a_provider_that_names_a_subject_but_no_email_stops_the_login(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    google_http.json_route("oauth2.googleapis.com/token", {"access_token": "at"})
    google_http.json_route("v1/userinfo", {"sub": "s", "email": ""})
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE


async def test_a_userinfo_endpoint_that_fails_stops_the_login(
    service: OAuthService, google_http: FakeHttp
) -> None:
    state, nonce = await start(service)
    google_http.json_route("oauth2.googleapis.com/token", {"access_token": "at"})
    google_http.fail_route("v1/userinfo")
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("google", code="c", state=state, nonce=nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE


# GitHub, which has no id_token and reports verification separately


async def test_github_reads_the_address_github_itself_vouches_for(
    service: OAuthService, http: FakeHttp
) -> None:
    state, nonce = await start(service, "github")
    http.json_route("login/oauth/access_token", {"access_token": "gho_x"})
    http.json_route("api.github.com/user", {"id": 42, "name": "Ada Lovelace"})
    http.route("user/emails", github_emails("ada@example.com"))
    await service.ahandle_callback("github", code="c", state=state, nonce=nonce)
    user = await service._user_store._aget("ada@example.com", column_name="email")
    assert user is not None
    assert user.first_name == "Ada"
    assert user.last_name == "Lovelace"


async def test_github_is_asked_for_its_profile_as_a_bearer(
    service: OAuthService, http: FakeHttp
) -> None:
    state, nonce = await start(service, "github")
    http.json_route("login/oauth/access_token", {"access_token": "gho_x"})
    http.json_route("api.github.com/user", {"id": 42, "email": "ada@example.com"})
    http.route("user/emails", github_emails("ada@example.com"))
    await service.ahandle_callback("github", code="c", state=state, nonce=nonce)
    headers = http.sent_to("api.github.com/user")["headers"]
    assert headers["Authorization"] == "Bearer gho_x"


async def test_a_github_account_with_no_published_address_stops_the_login(
    service: OAuthService, http: FakeHttp
) -> None:
    state, nonce = await start(service, "github")
    http.json_route("login/oauth/access_token", {"access_token": "gho_x"})
    http.json_route("api.github.com/user", {"id": 42, "email": None})
    http.route("user/emails", "[]")
    with pytest.raises(ErrorsException) as caught:
        await service.ahandle_callback("github", code="c", state=state, nonce=nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE


# Apple, which posts its callback and names the user exactly once


@pytest.fixture
def apple(make_identity_settings: Any, http: FakeHttp) -> OAuthService:
    """Apple enabled with a pre-generated client secret, and discovered."""
    enable_apple(make_identity_settings, http)
    return OAuthService()


async def test_apple_is_asked_to_post_its_callback(apple: OAuthService) -> None:
    url, _ = await apple.abuild_authorize_redirect("apple")
    assert query_of(url)["response_mode"] == "form_post"


async def test_apple_names_the_user_only_in_the_first_callback(
    apple: OAuthService,
) -> None:
    """The names arrive beside the code, once, and never again."""
    state, nonce = await start(apple, "apple")
    await apple.ahandle_callback(
        "apple",
        code="c",
        state=state,
        nonce=nonce,
        apple_user_json='{"name": {"firstName": "Ada", "lastName": "Lovelace"}}',
    )
    user = await apple._user_store._aget("person@example.com", column_name="email")
    assert user is not None
    assert (user.first_name, user.last_name) == ("Ada", "Lovelace")


async def test_apple_falls_back_to_the_posted_address_when_the_token_has_none(
    apple: OAuthService, http: FakeHttp
) -> None:
    state, nonce = await start(apple, "apple")
    http.json_route(
        "appleid.apple.com/auth/token",
        {"access_token": "at", "id_token": id_token(sub="apple-subject", email="")},
    )
    await apple.ahandle_callback(
        "apple",
        code="c",
        state=state,
        nonce=nonce,
        apple_user_json='{"email": "relay@privaterelay.appleid.com"}',
    )
    assert await apple._user_store._aget(
        "relay@privaterelay.appleid.com", column_name="email"
    )


async def test_a_posted_user_blob_that_is_not_json_is_ignored(
    apple: OAuthService,
) -> None:
    state, nonce = await start(apple, "apple")
    code = await apple.ahandle_callback(
        "apple", code="c", state=state, nonce=nonce, apple_user_json="{not json"
    )
    assert code


async def test_apple_without_an_id_token_cannot_name_anyone(
    apple: OAuthService, http: FakeHttp
) -> None:
    state, nonce = await start(apple, "apple")
    http.json_route("appleid.apple.com/auth/token", {"access_token": "at"})
    with pytest.raises(ErrorsException) as caught:
        await apple.ahandle_callback("apple", code="c", state=state, nonce=nonce)
    assert caught.value.error_enum is AuthErrors.OAUTH_PROFILE_INCOMPLETE
