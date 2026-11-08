"""How the Anthropic SDK's exceptions are read, as the LLM family.

Anthropic names more of its failures than most clients do — an overloaded
model, a deadline, a request too large — so the map is wider and worth
pinning class by class. There is no content-filter arm here on purpose: a
refusal arrives as a `stop_reason` on a successful response.
"""

from __future__ import annotations

import anthropic
import pytest
from anthropic import (
    AnthropicError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    CredentialsError,
    DeadlineExceededError,
    InternalServerError,
    NotFoundError,
    OverloadedError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
    ServiceUnavailableError,
    UnprocessableEntityError,
)

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.llm import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConflictError,
    LLMConnectionError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderRetryExhausted
from base.services.error_handling.llm.anthropic import AnthropicErrorHandler
from tests.fakes import subclasses_of
from tests.unit.error_handling.support import (
    build_sdk_error,
    domain_class_for,
    http_request,
    http_response,
)

STATUS_ERRORS: tuple[tuple[type[AnthropicError], int, FailureKind], ...] = (
    (RateLimitError, 429, FailureKind.RATE_LIMIT),
    (InternalServerError, 500, FailureKind.SERVER),
    (OverloadedError, 529, FailureKind.SERVER),
    (ServiceUnavailableError, 503, FailureKind.SERVER),
    (DeadlineExceededError, 504, FailureKind.TIMEOUT),
    (BadRequestError, 400, FailureKind.BAD_REQUEST),
    (UnprocessableEntityError, 422, FailureKind.BAD_REQUEST),
    (RequestTooLargeError, 413, FailureKind.BAD_REQUEST),
    (AuthenticationError, 401, FailureKind.AUTH),
    (PermissionDeniedError, 403, FailureKind.AUTH),
    (NotFoundError, 404, FailureKind.NOT_FOUND),
    (ConflictError, 409, FailureKind.CONFLICT),
)


def status_error(
    cls: type[AnthropicError],
    status: int,
    headers: dict[str, str] | None = None,
    message: str = "the provider said no",
) -> AnthropicError:
    """One of the SDK's HTTP errors, carrying a real response."""
    return cls(  # type: ignore[call-arg]
        message, response=http_response(status, headers), body=None
    )


@pytest.fixture
def handler() -> AnthropicErrorHandler:
    return AnthropicErrorHandler(retries=1, base_wait=0.0)


@pytest.mark.parametrize(("cls", "status", "kind"), STATUS_ERRORS)
def test_an_http_error_resolves_to_its_kind(
    handler: AnthropicErrorHandler,
    cls: type[AnthropicError],
    status: int,
    kind: FailureKind,
) -> None:
    assert handler.failure_kind(status_error(cls, status)) is kind


@pytest.mark.parametrize(("cls", "status", "kind"), STATUS_ERRORS)
def test_retryability_follows_the_kind_and_nothing_else(
    handler: AnthropicErrorHandler,
    cls: type[AnthropicError],
    status: int,
    kind: FailureKind,
) -> None:
    _, retryable = handler.classify(status_error(cls, status))
    assert retryable is (kind in RETRYABLE_FAILURES)


def test_an_overloaded_model_is_retried(handler: AnthropicErrorHandler) -> None:
    domain_cls, retryable = handler.classify(status_error(OverloadedError, 529))
    assert (domain_cls, retryable) == (LLMServerError, True)


def test_a_request_too_large_is_never_retried(
    handler: AnthropicErrorHandler,
) -> None:
    domain_cls, retryable = handler.classify(status_error(RequestTooLargeError, 413))
    assert (domain_cls, retryable) == (LLMBadRequestError, False)


def test_a_timeout_resolves_to_the_timeout_kind(
    handler: AnthropicErrorHandler,
) -> None:
    error = APITimeoutError(request=http_request())
    assert handler.classify(error) == (LLMTimeoutError, True)


def test_a_dropped_connection_resolves_to_the_connection_kind(
    handler: AnthropicErrorHandler,
) -> None:
    error = APIConnectionError(request=http_request())
    assert handler.classify(error) == (LLMConnectionError, True)


def test_a_rejected_credential_is_an_auth_failure(
    handler: AnthropicErrorHandler,
) -> None:
    assert handler.classify(CredentialsError("no key")) == (LLMAuthError, False)


def test_an_sdk_error_outside_the_map_is_unknown(
    handler: AnthropicErrorHandler,
) -> None:
    error = AnthropicError("something new")
    assert handler.failure_kind(error) is FailureKind.UNKNOWN
    assert handler.classify(error) == (LLMUnexpectedError, False)


def test_an_error_the_sdk_marks_retryable_is_retried(
    handler: AnthropicErrorHandler,
) -> None:
    """The SDK says so in its own docstring, so this handler honours it."""
    error = anthropic.RetryableError("middleware asked for another go")
    assert handler.failure_kind(error) is FailureKind.SERVER
    assert handler.classify(error)[1] is True


@pytest.mark.parametrize(
    ("cls", "status", "expected"),
    [
        (RateLimitError, 429, LLMRateLimitError),
        (InternalServerError, 500, LLMServerError),
        (AuthenticationError, 401, LLMAuthError),
        (NotFoundError, 404, LLMNotFoundError),
        (ConflictError, 409, LLMConflictError),
    ],
)
def test_each_kind_becomes_its_llm_class(
    handler: AnthropicErrorHandler,
    cls: type[AnthropicError],
    status: int,
    expected: type[Exception],
) -> None:
    assert handler.classify(status_error(cls, status))[0] is expected


def test_a_throttle_hint_is_read_off_the_response(
    handler: AnthropicErrorHandler,
) -> None:
    error = status_error(RateLimitError, 429, headers={"retry-after": "12"})
    assert handler.retry_after(error) == 12.0


def test_a_throttle_hint_that_is_not_a_number_is_ignored(
    handler: AnthropicErrorHandler,
) -> None:
    error = status_error(RateLimitError, 429, headers={"retry-after": "later"})
    assert handler.retry_after(error) is None


def test_no_hint_is_read_from_an_overloaded_model(
    handler: AnthropicErrorHandler,
) -> None:
    error = status_error(OverloadedError, 529, headers={"retry-after": "9"})
    assert handler.retry_after(error) is None


def test_a_wrapped_error_carries_the_status_and_request_id(
    handler: AnthropicErrorHandler,
) -> None:
    error = status_error(NotFoundError, 404, headers={"request-id": "req_1"})
    wrapped = handler.wrap(error, LLMNotFoundError)
    assert (wrapped.status_code, wrapped.request_id) == (404, "req_1")
    assert wrapped.provider_error is error


def test_an_exhausted_error_carries_the_attempts_and_the_status(
    handler: AnthropicErrorHandler,
) -> None:
    error = status_error(OverloadedError, 529, headers={"request-id": "req_2"})
    wrapped = handler.wrap_exhausted(error, 5)
    assert isinstance(wrapped, LLMServerError)
    assert (wrapped.attempts, wrapped.status_code) == (5, 529)
    assert "after 5 attempts" in str(wrapped)


def test_an_error_without_a_response_has_no_status(
    handler: AnthropicErrorHandler,
) -> None:
    wrapped = handler.wrap_exhausted(APITimeoutError(request=http_request()), 3)
    assert (wrapped.status_code, wrapped.request_id) == (None, None)


def test_something_the_sdk_never_raised_becomes_the_unexpected_error(
    handler: AnthropicErrorHandler,
) -> None:
    error = ValueError("the reply did not parse")
    wrapped = handler.wrap_unexpected(error)
    assert isinstance(wrapped, LLMUnexpectedError)
    assert wrapped.provider_error is error


def test_every_sdk_exception_maps_to_a_sane_kind() -> None:
    """Walks the SDK's whole tree so a class it adds later is covered too."""
    handler = AnthropicErrorHandler()
    covered = 0
    for cls in subclasses_of(anthropic.AnthropicError):
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
    assert covered >= 12


def test_the_sdk_base_class_is_what_the_handler_catches() -> None:
    assert AnthropicErrorHandler().provider_error_type is anthropic.AnthropicError
