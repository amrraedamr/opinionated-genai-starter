"""How the OpenAI SDK's exceptions are read, and what each domain makes of them.

The SDK half is shared: the LLM adapter and the embedding adapter classify
identically and differ only in which family they raise. The one divergence is
Azure's content filter, which arrives as an ordinary 400 and must not be
reported as one.
"""

from __future__ import annotations

import openai
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.embedding import (
    EmbeddingBadRequestError,
    EmbeddingError,
    EmbeddingNotFoundError,
    EmbeddingRateLimitError,
    EmbeddingUnexpectedError,
)
from base.models.view_models.exceptions.llm import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConflictError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderRetryExhausted
from base.services.error_handling.embedding.openai import OpenAIEmbeddingErrorHandler
from base.services.error_handling.llm.openai import OpenAIErrorHandler
from tests.fakes import subclasses_of
from tests.unit.error_handling.support import (
    build_sdk_error,
    domain_class_for,
    http_request,
    http_response,
)

STATUS_ERRORS: tuple[tuple[type[OpenAIError], int, FailureKind], ...] = (
    (RateLimitError, 429, FailureKind.RATE_LIMIT),
    (InternalServerError, 500, FailureKind.SERVER),
    (BadRequestError, 400, FailureKind.BAD_REQUEST),
    (UnprocessableEntityError, 422, FailureKind.BAD_REQUEST),
    (AuthenticationError, 401, FailureKind.AUTH),
    (PermissionDeniedError, 403, FailureKind.AUTH),
    (NotFoundError, 404, FailureKind.NOT_FOUND),
    (ConflictError, 409, FailureKind.CONFLICT),
)


def status_error(
    cls: type[OpenAIError],
    status: int,
    headers: dict[str, str] | None = None,
    body: object | None = None,
    message: str = "the provider said no",
) -> OpenAIError:
    """One of the SDK's HTTP errors, carrying a real response."""
    return cls(  # type: ignore[call-arg]
        message, response=http_response(status, headers), body=body
    )


@pytest.fixture
def handler() -> OpenAIErrorHandler:
    return OpenAIErrorHandler(retries=1, base_wait=0.0)


@pytest.mark.parametrize(("cls", "status", "kind"), STATUS_ERRORS)
def test_an_http_error_resolves_to_its_kind(
    handler: OpenAIErrorHandler,
    cls: type[OpenAIError],
    status: int,
    kind: FailureKind,
) -> None:
    assert handler.failure_kind(status_error(cls, status)) is kind


def test_a_timeout_resolves_to_the_timeout_kind(handler: OpenAIErrorHandler) -> None:
    assert handler.failure_kind(APITimeoutError(request=http_request())) is (
        FailureKind.TIMEOUT
    )


def test_a_dropped_connection_resolves_to_the_connection_kind(
    handler: OpenAIErrorHandler,
) -> None:
    error = APIConnectionError(request=http_request())
    assert handler.failure_kind(error) is FailureKind.CONNECTION


def test_an_sdk_error_outside_the_map_is_unknown(
    handler: OpenAIErrorHandler,
) -> None:
    assert handler.failure_kind(OpenAIError("something new")) is FailureKind.UNKNOWN
    domain_cls, retryable = handler.classify(OpenAIError("something new"))
    assert (domain_cls, retryable) == (LLMUnexpectedError, False)


@pytest.mark.parametrize(("cls", "status", "kind"), STATUS_ERRORS)
def test_retryability_follows_the_kind_and_nothing_else(
    handler: OpenAIErrorHandler,
    cls: type[OpenAIError],
    status: int,
    kind: FailureKind,
) -> None:
    _, retryable = handler.classify(status_error(cls, status))
    assert retryable is (kind in RETRYABLE_FAILURES)


@pytest.mark.parametrize(
    ("cls", "status", "expected"),
    [
        (RateLimitError, 429, LLMRateLimitError),
        (InternalServerError, 500, LLMServerError),
        (AuthenticationError, 401, LLMAuthError),
        (NotFoundError, 404, LLMNotFoundError),
        (ConflictError, 409, LLMConflictError),
        (UnprocessableEntityError, 422, LLMBadRequestError),
    ],
)
def test_the_llm_handler_raises_the_llm_family(
    handler: OpenAIErrorHandler,
    cls: type[OpenAIError],
    status: int,
    expected: type[Exception],
) -> None:
    domain_cls, _ = handler.classify(status_error(cls, status))
    assert domain_cls is expected


def test_the_embedding_handler_raises_the_embedding_family() -> None:
    embedding = OpenAIEmbeddingErrorHandler()
    domain_cls, retryable = embedding.classify(status_error(RateLimitError, 429))
    assert (domain_cls, retryable) == (EmbeddingRateLimitError, True)


def test_both_handlers_agree_on_the_kind_and_differ_only_on_the_family() -> None:
    llm = OpenAIErrorHandler()
    embedding = OpenAIEmbeddingErrorHandler()
    for cls, status, _ in STATUS_ERRORS:
        error = status_error(cls, status)
        assert llm.failure_kind(error) is embedding.failure_kind(error)
    missing = status_error(NotFoundError, 404)
    assert llm.classify(missing)[0] is LLMNotFoundError
    assert embedding.classify(missing)[0] is EmbeddingNotFoundError


def test_a_content_filter_rejection_is_not_an_ordinary_bad_request(
    handler: OpenAIErrorHandler,
) -> None:
    error = status_error(
        BadRequestError,
        400,
        body={"error": {"code": "content_filter", "message": "blocked"}},
    )
    domain_cls, retryable = handler.classify(error)
    assert (domain_cls, retryable) == (LLMContentFilterError, False)


def test_a_content_filter_rejection_named_only_in_the_message_is_recognised(
    handler: OpenAIErrorHandler,
) -> None:
    error = status_error(
        BadRequestError,
        400,
        message="The response was filtered due to the prompt triggering "
        "Azure OpenAI's content management policy.",
    )
    assert handler.classify(error)[0] is LLMContentFilterError


def test_a_content_filter_rejection_wraps_as_the_content_filter_error(
    handler: OpenAIErrorHandler,
) -> None:
    error = status_error(
        BadRequestError,
        400,
        headers={"x-request-id": "req-7"},
        body={"error": {"code": "content_filter"}},
    )
    wrapped = handler.wrap(error, LLMContentFilterError)
    assert isinstance(wrapped, LLMContentFilterError)
    assert (wrapped.status_code, wrapped.request_id) == (400, "req-7")
    assert wrapped.incomplete_details is None


def test_an_ordinary_bad_request_is_still_a_bad_request(
    handler: OpenAIErrorHandler,
) -> None:
    error = status_error(BadRequestError, 400, message="model is required")
    assert handler.classify(error)[0] is LLMBadRequestError


def test_the_embedding_handler_has_no_content_filter_arm() -> None:
    error = status_error(
        BadRequestError, 400, body={"error": {"code": "content_filter"}}
    )
    domain_cls, _ = OpenAIEmbeddingErrorHandler().classify(error)
    assert domain_cls is EmbeddingBadRequestError


def test_a_throttle_hint_is_read_off_the_response(
    handler: OpenAIErrorHandler,
) -> None:
    error = status_error(RateLimitError, 429, headers={"retry-after": "3.5"})
    assert handler.retry_after(error) == 3.5


def test_a_throttle_hint_that_is_not_a_number_is_ignored(
    handler: OpenAIErrorHandler,
) -> None:
    error = status_error(RateLimitError, 429, headers={"retry-after": "soon"})
    assert handler.retry_after(error) is None


def test_no_hint_is_read_from_anything_but_a_throttle(
    handler: OpenAIErrorHandler,
) -> None:
    error = status_error(InternalServerError, 500, headers={"retry-after": "3"})
    assert handler.retry_after(error) is None


def test_no_hint_is_read_when_the_header_is_absent(
    handler: OpenAIErrorHandler,
) -> None:
    assert handler.retry_after(status_error(RateLimitError, 429)) is None


def test_a_wrapped_error_carries_the_status_and_request_id(
    handler: OpenAIErrorHandler,
) -> None:
    error = status_error(NotFoundError, 404, headers={"x-request-id": "req-1"})
    wrapped = handler.wrap(error, LLMNotFoundError)
    assert isinstance(wrapped, LLMNotFoundError)
    assert (wrapped.status_code, wrapped.request_id) == (404, "req-1")
    assert wrapped.provider_error is error


def test_an_exhausted_error_carries_the_attempts_and_the_status(
    handler: OpenAIErrorHandler,
) -> None:
    error = status_error(InternalServerError, 503, headers={"x-request-id": "req-2"})
    wrapped = handler.wrap_exhausted(error, 4)
    assert isinstance(wrapped, LLMServerError)
    assert (wrapped.attempts, wrapped.status_code, wrapped.request_id) == (
        4,
        503,
        "req-2",
    )


def test_an_error_without_a_response_has_no_status(
    handler: OpenAIErrorHandler,
) -> None:
    wrapped = handler.wrap_exhausted(APITimeoutError(request=http_request()), 2)
    assert isinstance(wrapped, LLMTimeoutError)
    assert (wrapped.status_code, wrapped.request_id) == (None, None)
    assert wrapped.attempts == 2


def test_wrap_is_only_usable_for_a_class_that_reports_no_attempts(
    handler: OpenAIErrorHandler,
) -> None:
    """`wrap` never sees a retryable class, and could not build one if it did."""
    with pytest.raises(TypeError, match="attempts"):
        handler.wrap(status_error(InternalServerError, 500), LLMServerError)


def test_something_the_sdk_never_raised_becomes_the_unexpected_error(
    handler: OpenAIErrorHandler,
) -> None:
    error = ValueError("the reply did not parse")
    wrapped = handler.wrap_unexpected(error)
    assert isinstance(wrapped, LLMUnexpectedError)
    assert wrapped.provider_error is error


def test_the_embedding_handler_has_its_own_unexpected_error() -> None:
    wrapped = OpenAIEmbeddingErrorHandler().wrap_unexpected(ValueError("x"))
    assert isinstance(wrapped, EmbeddingUnexpectedError)
    assert isinstance(wrapped, EmbeddingError)
    assert not isinstance(wrapped, LLMError)


def test_a_connection_failure_retries_and_a_bad_request_does_not(
    handler: OpenAIErrorHandler,
) -> None:
    connection = APIConnectionError(request=http_request())
    assert handler.classify(connection) == (LLMConnectionError, True)
    assert handler.classify(status_error(BadRequestError, 400)) == (
        LLMBadRequestError,
        False,
    )


@pytest.mark.parametrize(
    "handler_cls", [OpenAIErrorHandler, OpenAIEmbeddingErrorHandler]
)
def test_every_sdk_exception_maps_to_a_sane_kind(handler_cls: type) -> None:
    """Walks the SDK's whole tree so a class it adds later is covered too."""
    instance = handler_cls()
    covered = 0
    for cls in subclasses_of(openai.OpenAIError):
        error = build_sdk_error(cls)
        if error is None:
            continue
        covered += 1
        kind = instance.failure_kind(error)
        assert isinstance(kind, FailureKind)
        domain_cls, retryable = instance.classify(error)
        assert retryable is (kind in RETRYABLE_FAILURES)
        if domain_cls is not LLMContentFilterError:
            assert domain_cls is domain_class_for(instance, kind)
        if retryable:
            assert issubclass(domain_cls, ProviderRetryExhausted)
            assert instance.wrap_exhausted(error, 2).attempts == 2
        else:
            assert isinstance(instance.wrap(error, domain_cls), domain_cls)
    assert covered >= 10


def test_the_sdk_base_class_is_what_the_handler_catches() -> None:
    assert OpenAIErrorHandler().provider_error_type is openai.OpenAIError
    assert OpenAIEmbeddingErrorHandler().provider_error_type is openai.OpenAIError
