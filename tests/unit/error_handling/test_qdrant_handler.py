"""How Qdrant's two exception shapes are read, as the vector search family.

Qdrant does not give each failure a class: either the server answered with a
status nobody expected, or it could not be reached at all. So the status does
all the work, and the boundary cases — 429, 5xx, 404, 409 — are what this file
pins.
"""

from __future__ import annotations

import httpx
import pytest
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
    UnexpectedResponse,
)

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.provider import ProviderRetryExhausted
from base.models.view_models.exceptions.vector_search import (
    VectorSearchAuthError,
    VectorSearchBadRequestError,
    VectorSearchConflictError,
    VectorSearchConnectionError,
    VectorSearchError,
    VectorSearchNotFoundError,
    VectorSearchRateLimitError,
    VectorSearchServerError,
    VectorSearchUnexpectedError,
)
from base.services.error_handling.vector_search.qdrant import QdrantErrorHandler
from tests.fakes import subclasses_of
from tests.unit.error_handling.support import build_sdk_error, domain_class_for

STATUS_KINDS: tuple[tuple[int, FailureKind], ...] = (
    (429, FailureKind.RATE_LIMIT),
    (500, FailureKind.SERVER),
    (503, FailureKind.SERVER),
    (401, FailureKind.AUTH),
    (403, FailureKind.AUTH),
    (404, FailureKind.NOT_FOUND),
    (409, FailureKind.CONFLICT),
    (400, FailureKind.BAD_REQUEST),
    (422, FailureKind.BAD_REQUEST),
)


def unexpected(
    status: int, headers: dict[str, str] | None = None
) -> UnexpectedResponse:
    """A reply Qdrant did not expect, carrying its status and headers."""
    return UnexpectedResponse(
        status_code=status,
        reason_phrase="Test",
        content=b'{"status":{"error":"no"}}',
        headers=httpx.Headers(headers or {}),
    )


@pytest.fixture
def handler() -> QdrantErrorHandler:
    return QdrantErrorHandler(retries=1, base_wait=0.0)


@pytest.mark.parametrize(("status", "kind"), STATUS_KINDS)
def test_a_status_resolves_to_its_kind(
    handler: QdrantErrorHandler, status: int, kind: FailureKind
) -> None:
    assert handler.failure_kind(unexpected(status)) is kind


@pytest.mark.parametrize(("status", "kind"), STATUS_KINDS)
def test_retryability_follows_the_kind_and_nothing_else(
    handler: QdrantErrorHandler, status: int, kind: FailureKind
) -> None:
    _, retryable = handler.classify(unexpected(status))
    assert retryable is (kind in RETRYABLE_FAILURES)


def test_a_server_that_could_not_be_reached_is_a_connection_failure(
    handler: QdrantErrorHandler,
) -> None:
    error = ResponseHandlingException(httpx.ConnectError("refused"))
    assert handler.classify(error) == (VectorSearchConnectionError, True)


def test_a_throttled_request_is_retried(handler: QdrantErrorHandler) -> None:
    assert handler.classify(unexpected(429)) == (VectorSearchRateLimitError, True)


def test_a_missing_collection_is_never_retried(
    handler: QdrantErrorHandler,
) -> None:
    assert handler.classify(unexpected(404)) == (VectorSearchNotFoundError, False)


def test_a_rejected_api_key_is_never_retried(handler: QdrantErrorHandler) -> None:
    assert handler.classify(unexpected(403)) == (VectorSearchAuthError, False)


def test_a_conflicting_write_is_never_retried(handler: QdrantErrorHandler) -> None:
    assert handler.classify(unexpected(409)) == (VectorSearchConflictError, False)


def test_a_malformed_query_is_never_retried(handler: QdrantErrorHandler) -> None:
    assert handler.classify(unexpected(400)) == (VectorSearchBadRequestError, False)


def test_a_failed_server_is_retried(handler: QdrantErrorHandler) -> None:
    assert handler.classify(unexpected(500)) == (VectorSearchServerError, True)


def test_a_status_below_four_hundred_is_unknown(
    handler: QdrantErrorHandler,
) -> None:
    assert handler.failure_kind(unexpected(304)) is FailureKind.UNKNOWN


def test_an_exception_shape_the_handler_does_not_know_is_unknown(
    handler: QdrantErrorHandler,
) -> None:
    error = ApiException("something new")
    assert handler.failure_kind(error) is FailureKind.UNKNOWN
    assert handler.classify(error) == (VectorSearchUnexpectedError, False)


@pytest.mark.parametrize("header", ["retry-after", "Retry-After"])
def test_a_shed_load_hint_is_read_whichever_case_it_arrives_in(
    handler: QdrantErrorHandler, header: str
) -> None:
    assert handler.retry_after(unexpected(429, {header: "6"})) == 6.0


def test_a_shed_load_hint_that_is_not_a_number_is_ignored(
    handler: QdrantErrorHandler,
) -> None:
    assert handler.retry_after(unexpected(429, {"retry-after": "soon"})) is None


def test_no_hint_is_read_when_the_header_is_absent(
    handler: QdrantErrorHandler,
) -> None:
    assert handler.retry_after(unexpected(429)) is None


def test_a_wrapped_error_carries_the_status(handler: QdrantErrorHandler) -> None:
    error = unexpected(404)
    wrapped = handler.wrap(error, VectorSearchNotFoundError)
    assert wrapped.status_code == 404
    assert wrapped.provider_error is error


def test_a_transport_failure_has_no_status(handler: QdrantErrorHandler) -> None:
    error = ResponseHandlingException(httpx.ConnectError("refused"))
    wrapped = handler.wrap_exhausted(error, 2)
    assert wrapped.status_code is None
    assert wrapped.attempts == 2


def test_an_exhausted_error_carries_the_attempts_and_the_status(
    handler: QdrantErrorHandler,
) -> None:
    wrapped = handler.wrap_exhausted(unexpected(503), 3)
    assert isinstance(wrapped, VectorSearchServerError)
    assert (wrapped.attempts, wrapped.status_code) == (3, 503)


def test_something_the_sdk_never_raised_becomes_the_unexpected_error(
    handler: QdrantErrorHandler,
) -> None:
    error = ValueError("the payload did not parse")
    wrapped = handler.wrap_unexpected(error)
    assert isinstance(wrapped, VectorSearchUnexpectedError)
    assert wrapped.provider_error is error


def test_every_sdk_exception_maps_to_a_sane_kind() -> None:
    """Walks the SDK's whole tree so a class it adds later is covered too."""
    handler = QdrantErrorHandler()
    covered = 0
    for cls in (ApiException, *subclasses_of(ApiException)):
        error = build_sdk_error(cls)
        if error is None:
            continue
        covered += 1
        kind = handler.failure_kind(error)
        assert isinstance(kind, FailureKind)
        domain_cls, retryable = handler.classify(error)
        assert retryable is (kind in RETRYABLE_FAILURES)
        assert domain_cls is domain_class_for(handler, kind)
        assert issubclass(domain_cls, VectorSearchError)
        if retryable:
            assert issubclass(domain_cls, ProviderRetryExhausted)
            assert handler.wrap_exhausted(error, 2).attempts == 2
        else:
            assert isinstance(handler.wrap(error, domain_cls), domain_cls)
    assert covered >= 3


def test_the_sdk_base_class_is_what_the_handler_catches() -> None:
    assert QdrantErrorHandler().provider_error_type is ApiException
