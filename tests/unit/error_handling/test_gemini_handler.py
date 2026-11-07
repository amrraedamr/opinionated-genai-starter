"""How Google GenAI failures are read, where only the HTTP status separates them.

The SDK reports everything as an `APIError` carrying a status, so there is no
exception type to read — a quota refusal and a malformed request are the same
class. This file pins the status boundaries, including the ones that are easy
to get wrong: 408 and 504 are timeouts, 409 is a conflict, and anything 5xx is
the server.
"""

from __future__ import annotations

import pytest
from google.genai import errors as genai_errors

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.llm import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConflictError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderRetryExhausted
from base.services.error_handling.llm.gemini import GeminiErrorHandler
from tests.fakes import subclasses_of
from tests.unit.error_handling.support import build_sdk_error, domain_class_for

STATUS_KINDS: tuple[tuple[int, FailureKind], ...] = (
    (429, FailureKind.RATE_LIMIT),
    (408, FailureKind.TIMEOUT),
    (504, FailureKind.TIMEOUT),
    (401, FailureKind.AUTH),
    (403, FailureKind.AUTH),
    (404, FailureKind.NOT_FOUND),
    (409, FailureKind.CONFLICT),
    (500, FailureKind.SERVER),
    (503, FailureKind.SERVER),
    (400, FailureKind.BAD_REQUEST),
    (422, FailureKind.BAD_REQUEST),
)


def api_error(
    code: int, status: str = "INVALID_ARGUMENT", message: str = "the model said no"
) -> genai_errors.APIError:
    """One of the SDK's errors, built from the payload it parses."""
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"message": message, "status": status}})


@pytest.fixture
def handler() -> GeminiErrorHandler:
    return GeminiErrorHandler(retries=1, base_wait=0.0)


@pytest.mark.parametrize(("code", "kind"), STATUS_KINDS)
def test_a_status_resolves_to_its_kind(
    handler: GeminiErrorHandler, code: int, kind: FailureKind
) -> None:
    assert handler.failure_kind(api_error(code)) is kind


@pytest.mark.parametrize(("code", "kind"), STATUS_KINDS)
def test_retryability_follows_the_kind_and_nothing_else(
    handler: GeminiErrorHandler, code: int, kind: FailureKind
) -> None:
    _, retryable = handler.classify(api_error(code))
    assert retryable is (kind in RETRYABLE_FAILURES)


def test_a_quota_refusal_is_a_throttle_and_is_retried(
    handler: GeminiErrorHandler,
) -> None:
    error = api_error(429, status="RESOURCE_EXHAUSTED", message="quota exceeded")
    assert handler.classify(error) == (LLMRateLimitError, True)


def test_an_unavailable_model_is_a_server_failure_and_is_retried(
    handler: GeminiErrorHandler,
) -> None:
    error = api_error(503, status="UNAVAILABLE", message="overloaded")
    assert handler.classify(error) == (LLMServerError, True)


def test_a_gateway_timeout_is_retried(handler: GeminiErrorHandler) -> None:
    assert handler.classify(api_error(504, status="DEADLINE_EXCEEDED")) == (
        LLMTimeoutError,
        True,
    )


def test_a_refused_request_is_never_retried(handler: GeminiErrorHandler) -> None:
    assert handler.classify(api_error(400)) == (LLMBadRequestError, False)


def test_a_rejected_key_is_never_retried(handler: GeminiErrorHandler) -> None:
    assert handler.classify(api_error(403, status="PERMISSION_DENIED")) == (
        LLMAuthError,
        False,
    )


def test_a_missing_model_is_never_retried(handler: GeminiErrorHandler) -> None:
    assert handler.classify(api_error(404, status="NOT_FOUND")) == (
        LLMNotFoundError,
        False,
    )


def test_a_conflict_is_never_retried(handler: GeminiErrorHandler) -> None:
    assert handler.classify(api_error(409, status="ABORTED")) == (
        LLMConflictError,
        False,
    )


def test_an_error_with_no_status_at_all_is_unknown(
    handler: GeminiErrorHandler,
) -> None:
    error = api_error(400)
    error.code = None  # type: ignore[assignment]
    assert handler.failure_kind(error) is FailureKind.UNKNOWN
    assert handler.classify(error) == (LLMUnexpectedError, False)


def test_a_status_the_sdk_should_never_send_is_unknown(
    handler: GeminiErrorHandler,
) -> None:
    error = api_error(400)
    error.code = 204
    assert handler.failure_kind(error) is FailureKind.UNKNOWN


def test_a_status_that_is_not_a_number_is_unknown(
    handler: GeminiErrorHandler,
) -> None:
    error = api_error(400)
    error.code = "429"  # type: ignore[assignment]
    assert handler.failure_kind(error) is FailureKind.UNKNOWN


def test_a_wrapped_error_names_the_providers_own_status(
    handler: GeminiErrorHandler,
) -> None:
    error = api_error(404, status="NOT_FOUND", message="model gone")
    wrapped = handler.wrap(error, LLMNotFoundError)
    assert wrapped.status_code == 404
    assert "NOT_FOUND: model gone" in str(wrapped)
    assert wrapped.provider_error is error


def test_an_exhausted_error_carries_the_attempts_and_the_status(
    handler: GeminiErrorHandler,
) -> None:
    error = api_error(429, status="RESOURCE_EXHAUSTED", message="quota")
    wrapped = handler.wrap_exhausted(error, 4)
    assert isinstance(wrapped, LLMRateLimitError)
    assert (wrapped.attempts, wrapped.status_code) == (4, 429)
    assert "RESOURCE_EXHAUSTED: quota" in str(wrapped)


def test_the_handler_offers_no_retry_hint(handler: GeminiErrorHandler) -> None:
    assert handler.retry_after(api_error(429, status="RESOURCE_EXHAUSTED")) is None


def test_something_the_sdk_never_raised_becomes_the_unexpected_error(
    handler: GeminiErrorHandler,
) -> None:
    error = ValueError("the reply did not parse")
    wrapped = handler.wrap_unexpected(error)
    assert isinstance(wrapped, LLMUnexpectedError)
    assert wrapped.provider_error is error


def test_a_client_error_and_a_server_error_are_read_the_same_way(
    handler: GeminiErrorHandler,
) -> None:
    client = genai_errors.ClientError(429, {"error": {"message": "quota"}})
    server = genai_errors.ServerError(503, {"error": {"message": "down"}})
    assert handler.failure_kind(client) is FailureKind.RATE_LIMIT
    assert handler.failure_kind(server) is FailureKind.SERVER


def test_every_sdk_exception_maps_to_a_sane_kind() -> None:
    """Walks the SDK's whole tree so a class it adds later is covered too."""
    handler = GeminiErrorHandler()
    covered = 0
    for cls in (genai_errors.APIError, *subclasses_of(genai_errors.APIError)):
        error = build_sdk_error(cls)
        if error is None:
            continue
        covered += 1
        kind = handler.failure_kind(error)
        assert isinstance(kind, FailureKind)
        domain_cls, retryable = handler.classify(error)
        assert retryable is (kind in RETRYABLE_FAILURES)
        assert domain_cls is domain_class_for(handler, kind)
        assert issubclass(domain_cls, LLMError)
        if retryable:
            assert issubclass(domain_cls, ProviderRetryExhausted)
            assert handler.wrap_exhausted(error, 2).attempts == 2
        else:
            assert isinstance(handler.wrap(error, domain_cls), domain_cls)
    assert covered >= 3


def test_the_sdk_base_class_is_what_the_handler_catches() -> None:
    assert GeminiErrorHandler().provider_error_type is genai_errors.APIError
