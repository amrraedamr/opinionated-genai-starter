"""The OAuth2 client-credentials grant this service uses on its own behalf.

Pins the request actually sent, the cache that keeps a burst of callers to one
token, the refresh skew that retires a token before a dependency does, and
the two ways providers want credentials presented. Nothing here touches the
network: the HTTP layer is replaced.
"""

import asyncio
import time
from typing import Any

import pytest

import base.clients.credentials.oauth2 as oauth2_module
from base.clients.credentials.oauth2 import OAuth2ClientCredentials
from base.models.definitions.errors import AuthErrors
from base.models.view_models.http import HttpResponse
from base.services.configuration.mixins import CredentialsSettingsMixin
from base.services.http import HttpRequestService
from base.tools.errors import ErrorsException
from tests.conftest import MakeSettings

TOKEN_URL = "https://idp.example.com/oauth2/token"


class FakeTokenEndpoint:
    """Answers the token endpoint, recording every request it was sent."""

    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, url: str, **kwargs: Any) -> HttpResponse:
        self.requests.append({"url": url, **kwargs})
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    @property
    def last(self) -> dict[str, Any]:
        assert self.requests, "nothing was sent"
        return self.requests[-1]

    @property
    def count(self) -> int:
        return len(self.requests)


class FakeClock:
    """The monotonic clock the cache reads, moved without sleeping."""

    def __init__(self) -> None:
        self.now = time.monotonic()

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(oauth2_module.time, "monotonic", lambda: fake.now)
    return fake


def _token(access_token: str = "at", **extra: Any) -> HttpResponse:
    return HttpResponse.ok(200, {"access_token": access_token, **extra})


def _endpoint(
    monkeypatch: pytest.MonkeyPatch, *responses: HttpResponse
) -> FakeTokenEndpoint:
    fake = FakeTokenEndpoint(*responses)
    monkeypatch.setattr(HttpRequestService, "acall_endpoint", fake)
    return fake


def _provider(**overrides: Any) -> OAuth2ClientCredentials:
    values: dict[str, Any] = {
        "token_url": TOKEN_URL,
        "client_id": "the-client",
        "client_secret": "the-secret",
    }
    values.update(overrides)
    return OAuth2ClientCredentials(**values)


def test_a_token_url_is_required() -> None:
    with pytest.raises(ValueError, match="token_url is required"):
        _provider(token_url="")


@pytest.mark.parametrize("missing", ["client_id", "client_secret"])
def test_a_client_id_and_secret_are_required(missing: str) -> None:
    with pytest.raises(ValueError, match="client_id and client_secret"):
        _provider(**{missing: ""})


def test_the_token_url_is_reachable_for_a_log_line() -> None:
    assert _provider().token_url == TOKEN_URL


async def test_the_grant_posts_a_client_credentials_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint(monkeypatch, _token("minted"))

    assert await _provider().aget_token() == "minted"
    assert endpoint.last["url"] == TOKEN_URL
    assert endpoint.last["method"] == "POST"
    assert endpoint.last["data"]["grant_type"] == "client_credentials"
    assert endpoint.last["headers"]["Accept"] == "application/json"
    assert (
        endpoint.last["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    )


async def test_a_scope_is_sent_only_when_one_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint(monkeypatch, _token())

    await _provider().aget_token()
    assert "scope" not in endpoint.last["data"]

    await _provider(scope="api://thing/.default").aget_token()
    assert endpoint.last["data"]["scope"] == "api://thing/.default"


async def test_basic_placement_sends_the_credentials_as_http_basic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint(monkeypatch, _token())

    await _provider(credential_placement="basic").aget_token()
    assert endpoint.last["auth"] == ("the-client", "the-secret")
    assert "client_id" not in endpoint.last["data"]
    assert "client_secret" not in endpoint.last["data"]


async def test_basic_placement_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint(monkeypatch, _token())
    await _provider().aget_token()
    assert endpoint.last["auth"] == ("the-client", "the-secret")


async def test_form_placement_sends_the_credentials_as_form_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint(monkeypatch, _token())

    await _provider(credential_placement="form").aget_token()
    assert endpoint.last["auth"] is None
    assert endpoint.last["data"]["client_id"] == "the-client"
    assert endpoint.last["data"]["client_secret"] == "the-secret"


async def test_extra_form_fields_are_sent_and_win_over_the_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider that wants `resource`, or its own `grant_type` spelling."""
    endpoint = _endpoint(monkeypatch, _token())

    await _provider(
        extra_form_fields={"resource": "api://thing", "grant_type": "custom"}
    ).aget_token()
    assert endpoint.last["data"]["resource"] == "api://thing"
    assert endpoint.last["data"]["grant_type"] == "custom"


async def test_extra_form_fields_are_copied_not_held_by_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint(monkeypatch, _token())
    fields = {"resource": "api://thing"}
    provider = _provider(extra_form_fields=fields)
    fields["resource"] = "api://something-else"

    await provider.aget_token()
    assert endpoint.last["data"]["resource"] == "api://thing"


async def test_a_cached_token_is_reused_rather_than_reminted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint(monkeypatch, _token("first", expires_in=3600))
    provider = _provider()

    assert await provider.aget_token() == "first"
    assert await provider.aget_token() == "first"
    assert endpoint.count == 1


async def test_a_burst_of_callers_mints_one_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint(monkeypatch, _token("first", expires_in=3600))
    provider = _provider()

    tokens = await asyncio.gather(*(provider.aget_token() for _ in range(10)))
    assert tokens == ["first"] * 10
    assert endpoint.count == 1


async def test_a_token_near_expiry_is_reminted(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    endpoint = _endpoint(
        monkeypatch,
        _token("first", expires_in=120),
        _token("second", expires_in=120),
    )
    provider = _provider(refresh_skew_seconds=30)

    assert await provider.aget_token() == "first"
    clock.advance(91)
    assert await provider.aget_token() == "second"
    assert endpoint.count == 2


async def test_a_token_outside_the_skew_window_is_still_served(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    endpoint = _endpoint(monkeypatch, _token("first", expires_in=120))
    provider = _provider(refresh_skew_seconds=30)

    assert await provider.aget_token() == "first"
    clock.advance(89)
    assert await provider.aget_token() == "first"
    assert endpoint.count == 1


async def test_a_skew_above_the_issued_lifetime_never_serves_from_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented consequence of the knob, pinned so it is not a surprise."""
    endpoint = _endpoint(monkeypatch, _token("minted", expires_in=60))
    provider = _provider(refresh_skew_seconds=600)

    await provider.aget_token()
    await provider.aget_token()
    assert endpoint.count == 2


async def test_a_negative_skew_is_clamped_to_zero(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    endpoint = _endpoint(monkeypatch, _token("minted", expires_in=60))
    provider = _provider(refresh_skew_seconds=-100)

    await provider.aget_token()
    clock.advance(59)
    await provider.aget_token()
    assert endpoint.count == 1


async def test_a_response_without_an_expiry_falls_back_to_the_default_lifetime(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    endpoint = _endpoint(monkeypatch, _token("minted"))
    provider = _provider(default_lifetime_seconds=100, refresh_skew_seconds=0)

    await provider.aget_token()
    clock.advance(99)
    await provider.aget_token()
    assert endpoint.count == 1

    clock.advance(2)
    await provider.aget_token()
    assert endpoint.count == 2


def test_the_documented_defaults_are_an_hour_and_a_minute() -> None:
    assert OAuth2ClientCredentials.DEFAULT_LIFETIME_SECONDS == 3600.0
    assert OAuth2ClientCredentials.DEFAULT_REFRESH_SKEW_SECONDS == 60.0


async def test_invalidate_drops_the_cached_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For a dependency that rejects a token this side still believes in."""
    endpoint = _endpoint(
        monkeypatch,
        _token("first", expires_in=3600),
        _token("second", expires_in=3600),
    )
    provider = _provider()

    assert await provider.aget_token() == "first"
    provider.invalidate()
    assert await provider.aget_token() == "second"
    assert endpoint.count == 2


async def test_a_failed_token_endpoint_is_reported_as_an_outbound_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _endpoint(
        monkeypatch,
        HttpResponse.fail(RuntimeError("boom"), 401, error_detail="invalid_client"),
    )
    with pytest.raises(ErrorsException) as caught:
        await _provider().aget_token()
    assert caught.value.error_enum is AuthErrors.OUTBOUND_TOKEN_FAILED


async def test_a_token_endpoint_returning_no_access_token_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _endpoint(monkeypatch, HttpResponse.ok(200, {"token_type": "Bearer"}))
    with pytest.raises(ErrorsException) as caught:
        await _provider().aget_token()
    assert caught.value.error_enum is AuthErrors.OUTBOUND_TOKEN_FAILED


async def test_a_token_endpoint_returning_a_non_object_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _endpoint(monkeypatch, HttpResponse(success=True, status_code=200, data=None))
    with pytest.raises(ErrorsException) as caught:
        await _provider().aget_token()
    assert caught.value.error_enum is AuthErrors.OUTBOUND_TOKEN_FAILED


async def test_a_failure_leaves_nothing_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint(
        monkeypatch,
        HttpResponse.fail(RuntimeError("boom"), 500),
        _token("recovered", expires_in=3600),
    )
    provider = _provider()

    with pytest.raises(ErrorsException):
        await provider.aget_token()
    assert await provider.aget_token() == "recovered"
    assert endpoint.count == 2


async def test_the_lifetime_a_provider_reports_is_what_is_cached(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    endpoint = _endpoint(
        monkeypatch,
        _token("short", expires_in=10),
        _token("next", expires_in=10),
    )
    provider = _provider(refresh_skew_seconds=0, default_lifetime_seconds=10_000)

    assert await provider.aget_token() == "short"
    clock.advance(11)
    assert await provider.aget_token() == "next"
    assert endpoint.count == 2


def test_the_credential_placement_setting_names_the_two_supported_values() -> None:
    field = CredentialsSettingsMixin.model_fields["outbound_credential_placement"]
    assert field.default == "basic"
    assert (
        OAuth2ClientCredentials(
            token_url=TOKEN_URL, client_id="a", client_secret="b"
        )._credential_placement
        == "basic"
    )


async def test_the_refresh_skew_setting_reaches_the_provider(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    """`OUTBOUND_TOKEN_REFRESH_SKEW_SECONDS` is the knob behind the window."""
    from base.capabilities.credentials import OutboundCredentials

    make_settings(
        CredentialsSettingsMixin,
        outbound_token_url=TOKEN_URL,
        outbound_client_id="the-client",
        outbound_client_secret="the-secret",
        outbound_scope="api://thing/.default",
        outbound_credential_placement="form",
        outbound_token_refresh_skew_seconds=45,
    )
    endpoint = _endpoint(monkeypatch, _token("minted", expires_in=100))
    provider = OutboundCredentials().provider

    assert isinstance(provider, OAuth2ClientCredentials)
    assert provider._refresh_skew_seconds == 45.0
    assert provider._credential_placement == "form"

    await provider.aget_token()
    clock.advance(56)
    await provider.aget_token()
    assert endpoint.count == 2
