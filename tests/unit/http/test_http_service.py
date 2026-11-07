"""`HttpRequestService`: the request it builds, and what it returns on failure.

Nothing here leaves the process: the async path runs on an `httpx.MockTransport`
and the sync path on a stubbed `requests.Session`. The asymmetry between the two
is deliberate and is pinned — an `httpx.Auth` hook works on one and is refused
by the other, with a message saying which to use.
"""

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import requests

from base.models.view_models.http import HttpResponse
from base.services.common.meta import Singleton, purge_all
from base.services.http import HttpRequestService
from tests.fakes import no_sleep


@pytest.fixture(autouse=True)
def _purge_instances() -> Iterator[None]:
    """The service is a singleton holding a session and an async client."""
    purge_all()
    yield
    purge_all()


@pytest.fixture
def service() -> HttpRequestService:
    return HttpRequestService()


def mock_async(service: HttpRequestService, handler: Any) -> list[httpx.Request]:
    """Point the async client at a transport, and record what it was sent."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    service._async_client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return seen


def json_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"id": 7},
        headers={"Content-Type": "application/json", "X-Trace": "abc"},
    )


class Bearer(httpx.Auth):
    """The shape a caller actually passes: a hook that signs each request."""

    def auth_flow(self, request: httpx.Request) -> Any:
        request.headers["Authorization"] = "Bearer token-123"
        yield request


def test_the_service_is_one_object_for_the_whole_process() -> None:
    assert HttpRequestService() is HttpRequestService()
    assert type(HttpRequestService) is Singleton


def test_both_clients_are_built_up_front(service: HttpRequestService) -> None:
    assert isinstance(service.sync_client, requests.Session)
    assert isinstance(service.async_client, httpx.AsyncClient)


def test_the_default_timeout_is_shorter_than_the_transport_timeout(
    service: HttpRequestService,
) -> None:
    assert service._default_api_timeout == 10.0
    assert HttpRequestService._MAX_RETRIES == 3
    assert HttpRequestService._BASE_BACKOFF_SECONDS == 0.5


async def test_a_json_reply_is_parsed_into_the_response_envelope(
    service: HttpRequestService,
) -> None:
    mock_async(service, json_ok)
    response = await service.acall_endpoint("https://api.example/widgets")
    assert isinstance(response, HttpResponse)
    assert response.success is True
    assert response.failed is False
    assert response.status_code == 200
    assert response.data == {"id": 7}


async def test_response_headers_are_lowercased_for_case_insensitive_lookup(
    service: HttpRequestService,
) -> None:
    mock_async(service, json_ok)
    response = await service.acall_endpoint("https://api.example/widgets")
    assert response.headers["x-trace"] == "abc"
    assert "X-Trace" not in response.headers


async def test_a_non_json_reply_is_returned_as_text_beside_its_url(
    service: HttpRequestService,
) -> None:
    mock_async(
        service,
        lambda request: httpx.Response(
            200, text="plain body", headers={"Content-Type": "text/plain"}
        ),
    )
    response = await service.acall_endpoint("https://api.example/widgets")
    assert response.data == {
        "url": "https://api.example/widgets",
        "text": "plain body",
    }


async def test_a_graphql_content_type_is_treated_as_json(
    service: HttpRequestService,
) -> None:
    mock_async(
        service,
        lambda request: httpx.Response(
            200,
            json={"data": {"me": 1}},
            headers={"Content-Type": "application/graphql-response+json"},
        ),
    )
    response = await service.acall_endpoint("https://api.example/graphql")
    assert response.data == {"data": {"me": 1}}


async def test_raw_mode_returns_the_bytes_rather_than_the_envelope(
    service: HttpRequestService,
) -> None:
    mock_async(service, lambda request: httpx.Response(200, content=b"\x00\x01"))
    assert (
        await service.acall_endpoint("https://api.example/f", raw=True) == b"\x00\x01"
    )


async def test_the_method_query_and_body_are_sent_as_given(
    service: HttpRequestService,
) -> None:
    seen = mock_async(service, json_ok)
    await service.acall_endpoint(
        "https://api.example/widgets",
        method="POST",
        query_params={"page": "2"},
        body={"name": "widget"},
        headers={"X-Client": "tests"},
    )
    request = seen[-1]
    assert request.method == "POST"
    assert request.url.params["page"] == "2"
    assert request.headers["x-client"] == "tests"
    assert request.read() == b'{"name":"widget"}'


async def test_an_httpx_auth_hook_signs_the_async_request(
    service: HttpRequestService,
) -> None:
    seen = mock_async(service, json_ok)
    await service.acall_endpoint("https://api.example/widgets", auth=Bearer())
    assert seen[-1].headers["authorization"] == "Bearer token-123"


async def test_a_basic_auth_pair_also_works_on_the_async_path(
    service: HttpRequestService,
) -> None:
    seen = mock_async(service, json_ok)
    await service.acall_endpoint("https://api.example/widgets", auth=("user", "pass"))
    assert seen[-1].headers["authorization"].startswith("Basic ")


def test_the_sync_path_refuses_an_httpx_auth_hook_and_says_which_to_use(
    service: HttpRequestService,
) -> None:
    with pytest.raises(TypeError, match="use acall_endpoint"):
        service.call_endpoint("https://api.example/widgets", auth=Bearer())


def test_the_sync_path_accepts_a_username_and_password_pair(
    service: HttpRequestService,
) -> None:
    assert service._sync_auth(("user", "pass")) == ("user", "pass")
    assert service._sync_auth(None) is None


async def test_a_failing_status_becomes_a_failed_envelope_with_its_detail(
    service: HttpRequestService,
) -> None:
    mock_async(
        service,
        lambda request: httpx.Response(
            422,
            json={"detail": "name is required"},
            headers={"Content-Type": "application/json", "X-Req": "r1"},
        ),
    )
    response = await service.acall_endpoint("https://api.example/widgets")
    assert response.success is False
    assert response.status_code == 422
    assert response.error_detail == {"detail": "name is required"}
    assert response.headers["x-req"] == "r1"
    assert isinstance(response.error, httpx.HTTPStatusError)


async def test_a_failing_status_with_a_non_json_body_keeps_the_text(
    service: HttpRequestService,
) -> None:
    mock_async(service, lambda request: httpx.Response(503, text="upstream down"))
    response = await service.acall_endpoint("https://api.example/widgets")
    assert response.error_detail == "upstream down"


async def test_a_failing_status_raises_in_raw_mode(
    service: HttpRequestService,
) -> None:
    mock_async(service, lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(RuntimeError, match="Request failed with status 500"):
        await service.acall_endpoint("https://api.example/widgets", raw=True)


async def test_a_transport_failure_becomes_a_failed_envelope(
    service: HttpRequestService,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    mock_async(service, refuse)
    response = await service.acall_endpoint("https://api.example/widgets")
    assert response.success is False
    assert response.status_code is None
    assert isinstance(response.error, httpx.ConnectError)


async def test_a_transport_failure_raises_in_raw_mode(
    service: HttpRequestService,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    mock_async(service, refuse)
    with pytest.raises(RuntimeError, match="Request failed"):
        await service.acall_endpoint("https://api.example/widgets", raw=True)


async def test_a_timeout_is_retried_up_to_the_count_asked_for(
    service: HttpRequestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_sleep(monkeypatch)
    attempts = {"count": 0}

    def time_out(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ReadTimeout("too slow", request=request)

    mock_async(service, time_out)
    response = await service.acall_endpoint(
        "https://api.example/widgets", retry_count=2
    )
    assert attempts["count"] == 3
    assert response.success is False
    assert isinstance(response.error, httpx.ReadTimeout)


async def test_a_timeout_that_clears_on_a_later_attempt_succeeds(
    service: HttpRequestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_sleep(monkeypatch)
    attempts = {"count": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.ReadTimeout("too slow", request=request)
        return json_ok(request)

    mock_async(service, flaky)
    response = await service.acall_endpoint("https://api.example/widgets")
    assert response.success is True
    assert attempts["count"] == 3


async def test_a_failing_status_is_not_retried(
    service: HttpRequestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a timeout is worth repeating; a 500 would answer the same way."""
    no_sleep(monkeypatch)
    attempts = {"count": 0}

    def fail(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(500, text="boom")

    mock_async(service, fail)
    await service.acall_endpoint("https://api.example/widgets", retry_count=3)
    assert attempts["count"] == 1


async def test_retries_can_be_turned_off_entirely(
    service: HttpRequestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_sleep(monkeypatch)
    attempts = {"count": 0}

    def time_out(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ReadTimeout("too slow", request=request)

    mock_async(service, time_out)
    await service.acall_endpoint("https://api.example/widgets", retry_count=0)
    assert attempts["count"] == 1


async def test_an_exhausted_timeout_raises_in_raw_mode(
    service: HttpRequestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_sleep(monkeypatch)

    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    mock_async(service, time_out)
    with pytest.raises(RuntimeError, match="timed out after 1 attempts"):
        await service.acall_endpoint(
            "https://api.example/widgets", retry_count=0, raw=True
        )


async def test_the_caller_can_shorten_the_timeout_for_one_request(
    service: HttpRequestService,
) -> None:
    seen = mock_async(service, json_ok)
    await service.acall_endpoint("https://api.example/widgets", timeout=0.25)
    assert seen[-1].extensions["timeout"]["read"] == 0.25


async def test_a_request_with_no_timeout_uses_the_service_default(
    service: HttpRequestService,
) -> None:
    seen = mock_async(service, json_ok)
    await service.acall_endpoint("https://api.example/widgets")
    assert seen[-1].extensions["timeout"]["read"] == 10.0


async def test_a_redirect_is_followed_on_the_async_path(
    service: HttpRequestService,
) -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(307, headers={"Location": "https://api.example/new"})
        return json_ok(request)

    seen = mock_async(service, redirect)
    response = await service.acall_endpoint("https://api.example/old")
    assert response.success is True
    assert [str(r.url) for r in seen] == [
        "https://api.example/old",
        "https://api.example/new",
    ]


class FakeSyncResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
        url: str = "https://api.example/widgets",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}
        self.url = url
        self.content = text.encode()

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"{self.status_code}")
            error.response = self  # type: ignore[assignment]
            raise error


class FakeSession:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if callable(self.outcome):
            return self.outcome(len(self.calls))
        return self.outcome


def test_the_sync_path_sends_the_method_body_and_timeout_it_was_given(
    service: HttpRequestService,
) -> None:
    session = FakeSession(
        FakeSyncResponse(
            200, payload={"id": 7}, headers={"Content-Type": "application/json"}
        )
    )
    service._sync_client = session  # type: ignore[assignment]
    response = service.call_endpoint(
        "https://api.example/widgets", method="PUT", body={"name": "w"}, timeout=2.5
    )
    sent = session.calls[-1]
    assert sent["method"] == "PUT"
    assert sent["json"] == {"name": "w"}
    assert sent["timeout"] == 2.5
    assert response.data == {"id": 7}


def test_the_sync_path_falls_back_to_the_service_timeout(
    service: HttpRequestService,
) -> None:
    session = FakeSession(FakeSyncResponse(200, text="ok"))
    service._sync_client = session  # type: ignore[assignment]
    service.call_endpoint("https://api.example/widgets")
    assert session.calls[-1]["timeout"] == 10.0


def test_a_sync_failing_status_carries_its_detail_and_headers(
    service: HttpRequestService,
) -> None:
    failing = FakeSyncResponse(
        409, payload={"detail": "exists"}, headers={"X-Req": "r2"}
    )
    service._sync_client = FakeSession(failing)  # type: ignore[assignment]
    response = service.call_endpoint("https://api.example/widgets")
    assert response.success is False
    assert response.status_code == 409
    assert response.error_detail == {"detail": "exists"}
    assert response.headers["x-req"] == "r2"


def test_a_sync_request_error_without_a_response_still_fails_cleanly(
    service: HttpRequestService,
) -> None:
    service._sync_client = FakeSession(  # type: ignore[assignment]
        requests.exceptions.ConnectionError("no route")
    )
    response = service.call_endpoint("https://api.example/widgets")
    assert response.success is False
    assert response.status_code is None
    assert response.error_detail is None


def test_a_sync_timeout_is_retried_then_reported(
    service: HttpRequestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_sleep(monkeypatch)
    session = FakeSession(requests.exceptions.Timeout("too slow"))
    service._sync_client = session  # type: ignore[assignment]
    response = service.call_endpoint("https://api.example/widgets", retry_count=2)
    assert len(session.calls) == 3
    assert response.success is False
    assert isinstance(response.error, requests.exceptions.Timeout)


def test_a_sync_timeout_that_clears_succeeds(
    service: HttpRequestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_sleep(monkeypatch)

    def outcome(attempt: int) -> Any:
        if attempt < 2:
            raise requests.exceptions.Timeout("too slow")
        return FakeSyncResponse(200, text="ok")

    service._sync_client = FakeSession(outcome)  # type: ignore[assignment]
    assert service.call_endpoint("https://api.example/widgets").success is True


def test_a_sync_exhausted_timeout_raises_in_raw_mode(
    service: HttpRequestService, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_sleep(monkeypatch)
    service._sync_client = FakeSession(  # type: ignore[assignment]
        requests.exceptions.Timeout("too slow")
    )
    with pytest.raises(RuntimeError, match="timed out after 1 attempts"):
        service.call_endpoint("https://api.example/w", retry_count=0, raw=True)


def test_a_sync_request_error_raises_in_raw_mode(
    service: HttpRequestService,
) -> None:
    service._sync_client = FakeSession(  # type: ignore[assignment]
        requests.exceptions.ConnectionError("no route")
    )
    with pytest.raises(RuntimeError, match="Request failed"):
        service.call_endpoint("https://api.example/w", raw=True)


def test_a_successful_envelope_reports_no_error() -> None:
    response = HttpResponse.ok(200, {"a": 1}, {"x": "y"})
    assert response.success is True
    assert response.failed is False
    assert response.error is None


def test_a_failed_envelope_defaults_its_headers_to_empty() -> None:
    response = HttpResponse.fail(RuntimeError("x"))
    assert response.failed is True
    assert response.headers == {}
    assert response.status_code is None
