"""What the error-handling tests need before they can assert anything.

Two things live here. A stub provider — an exception family, a domain family
and a handler joining them — which lets the abstract machinery be tested
without an SDK deciding what a failure means. And builders for the real SDK
exceptions, because most of them refuse to be constructed from a string alone
and a test that cannot build one cannot check how it classifies.
"""

from __future__ import annotations

import sys
from typing import Any

import httpx

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.services.error_handling import AbstractErrorHandler
from tests.fakes import try_build

REQUEST_URL = "https://provider.test/v1/resource"


class StubProviderError(Exception):
    """The client exception a stub handler understands, carrying its verdict."""

    def __init__(
        self,
        kind: FailureKind = FailureKind.SERVER,
        message: str = "boom",
        hint: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.hint = hint


class StubUnrelatedError(Exception):
    """Something the stub client never promised to raise."""


class StubError(ProviderError):
    """Base of the stub domain family."""


class StubBadRequestError(StubError):
    """The request itself was wrong."""


class StubAuthError(StubError):
    """The credential was rejected."""


class StubNotFoundError(StubError):
    """The resource is not there."""


class StubConflictError(StubError):
    """The write collided with another."""


class StubUnexpectedError(StubError):
    """Nothing the handler anticipated."""


class StubRetryExhausted(StubError, ProviderRetryExhausted):
    """A transient stub failure that outlived its retries."""


class StubRateLimitError(StubRetryExhausted):
    """Throttled."""


class StubServerError(StubRetryExhausted):
    """The far end failed."""


class StubTimeoutError(StubRetryExhausted):
    """The call timed out."""


class StubConnectionError(StubRetryExhausted):
    """The far end was unreachable."""


STUB_ERRORS: dict[FailureKind, type[StubError]] = {
    FailureKind.RATE_LIMIT: StubRateLimitError,
    FailureKind.SERVER: StubServerError,
    FailureKind.TIMEOUT: StubTimeoutError,
    FailureKind.CONNECTION: StubConnectionError,
    FailureKind.BAD_REQUEST: StubBadRequestError,
    FailureKind.AUTH: StubAuthError,
    FailureKind.NOT_FOUND: StubNotFoundError,
    FailureKind.CONFLICT: StubConflictError,
    FailureKind.UNKNOWN: StubUnexpectedError,
}


class StubHandler(AbstractErrorHandler[StubProviderError]):
    """A handler whose classification is whatever the test put on the error."""

    @property
    def provider_error_type(self) -> type[StubProviderError]:
        return StubProviderError

    def failure_kind(self, error: StubProviderError) -> FailureKind:
        return error.kind

    def classify(self, error: StubProviderError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return STUB_ERRORS[kind], kind in RETRYABLE_FAILURES

    def retry_after(self, error: StubProviderError) -> float | None:
        return error.hint

    def wrap(self, error: StubProviderError, domain_cls: type[Exception]) -> Exception:
        return domain_cls(str(error), provider_error=error)

    def wrap_exhausted(self, error: StubProviderError, attempts: int) -> Exception:
        domain_cls, _ = self.classify(error)
        assert issubclass(domain_cls, ProviderRetryExhausted), (
            f"Retryable error {type(error).__name__} mapped to non-retry-exhausted "
            f"domain class {domain_cls.__name__}"
        )
        return domain_cls(
            f"{domain_cls.__name__} after {attempts} attempts: {error}",
            attempts=attempts,
            provider_error=error,
        )

    def wrap_unexpected(self, error: Exception) -> Exception:
        return StubUnexpectedError.from_provider(error)


class MisMappedHandler(StubHandler):
    """Maps a retryable kind to a class that cannot report attempts.

    The mistake `wrap_exhausted`'s assertion exists to catch.
    """

    def classify(self, error: StubProviderError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        if kind is FailureKind.SERVER:
            return StubBadRequestError, True
        return super().classify(error)


def record_waits(monkeypatch: Any) -> list[float]:
    """Capture every backoff instead of serving it, and return the list."""
    import asyncio
    import time

    waits: list[float] = []

    async def _asleep(seconds: float) -> None:
        waits.append(seconds)

    def _sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _asleep)
    monkeypatch.setattr(time, "sleep", _sleep)
    return waits


def http_request() -> httpx.Request:
    """A request object the httpx-based SDKs need to build an exception."""
    return httpx.Request("POST", REQUEST_URL)


def http_response(
    status: int = 500, headers: dict[str, str] | None = None
) -> httpx.Response:
    """A response object carrying a status and whatever headers a test needs."""
    return httpx.Response(status, headers=headers or {}, request=http_request())


class AzureResponse:
    """The shape `azure.core` reads a status and headers off.

    The SDK's own response wraps a transport reply that a unit test has no way
    to produce, and only these four attributes are ever touched.
    """

    def __init__(
        self, status: int = 500, headers: dict[str, str] | None = None
    ) -> None:
        self.status_code = status
        self.headers = headers or {}
        self.reason = "Test"
        self.content_type = "application/json"

    def text(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return "{}"


def build_sdk_error(cls: type) -> Exception | None:
    """Best-effort construction of an SDK exception, SDK shapes included.

    `tests.fakes.try_build` covers plain signatures; the HTTP clients want a
    response or a request object, and the database drivers want a statement
    and an original error, so those shapes are tried here too.
    """
    built = try_build(cls)
    if built is not None:
        return built

    request = http_request()
    response = http_response()
    shapes: tuple[Any, ...] = (
        lambda: cls("boom", response=response, body=None),
        lambda: cls(request=request),
        lambda: cls("boom", request=request),
        lambda: cls(message="boom", request=request),
        lambda: cls(status_code=500, headers={}, body=None),
        lambda: cls(500, "boom", b"{}", httpx.Headers({})),
        lambda: cls(500, {"error": {"message": "boom", "status": "INTERNAL"}}),
        lambda: cls("boom", {}, Exception("driver said no")),
        lambda: cls(message="boom", response=AzureResponse()),
        lambda: cls(response=AzureResponse()),
        lambda: cls({"writeErrors": [], "writeConcernErrors": []}),
        lambda: cls("1234 127.0.0.1:6379"),
    )
    for shape in shapes:
        try:
            candidate = shape()
        except Exception:
            continue
        if isinstance(candidate, BaseException):
            return candidate  # type: ignore[return-value]
    return None


def domain_class_for(handler: AbstractErrorHandler[Any], kind: FailureKind) -> type:
    """The domain class a handler turns a kind into.

    Handlers reach it two ways — an `error_for` hook when the SDK half is
    shared, a module-level map when it is not — and a contract test should not
    care which.
    """
    error_for = getattr(handler, "error_for", None)
    if error_for is not None:
        return error_for(kind)
    module = sys.modules[type(handler).__module__]
    return module._ERROR_MAP[kind]
