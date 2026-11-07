"""How Cohere's exceptions are read, as the rerank family.

Cohere gives every status its own class, so the map is a straight
correspondence — which is exactly the kind of table that drifts silently when
the SDK adds a status. `ClientClosedRequestError` is the odd one: a 499 is the
client having hung up, which reads as a connection failure and is retried.
"""

from __future__ import annotations

import pytest
from cohere.core import ApiError
from cohere.errors import (
    BadRequestError,
    ClientClosedRequestError,
    ForbiddenError,
    GatewayTimeoutError,
    InternalServerError,
    InvalidTokenError,
    NotFoundError,
    ServiceUnavailableError,
    TooManyRequestsError,
    UnauthorizedError,
    UnprocessableEntityError,
)
from cohere.errors import (
    NotImplementedError as CohereNotImplementedError,
)

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.provider import ProviderRetryExhausted
from base.models.view_models.exceptions.rerank import (
    RerankAuthError,
    RerankBadRequestError,
    RerankConnectionError,
    RerankError,
    RerankNotFoundError,
    RerankRateLimitError,
    RerankServerError,
    RerankTimeoutError,
    RerankUnexpectedError,
)
from base.services.error_handling.rerank.cohere import CohereErrorHandler
from tests.fakes import subclasses_of
from tests.unit.error_handling.support import build_sdk_error, domain_class_for

MAPPED: tuple[tuple[type[ApiError], int, FailureKind, type[Exception]], ...] = (
    (TooManyRequestsError, 429, FailureKind.RATE_LIMIT, RerankRateLimitError),
    (InternalServerError, 500, FailureKind.SERVER, RerankServerError),
    (ServiceUnavailableError, 503, FailureKind.SERVER, RerankServerError),
    (GatewayTimeoutError, 504, FailureKind.TIMEOUT, RerankTimeoutError),
    (ClientClosedRequestError, 499, FailureKind.CONNECTION, RerankConnectionError),
    (BadRequestError, 400, FailureKind.BAD_REQUEST, RerankBadRequestError),
    (UnprocessableEntityError, 422, FailureKind.BAD_REQUEST, RerankBadRequestError),
    (CohereNotImplementedError, 501, FailureKind.BAD_REQUEST, RerankBadRequestError),
    (UnauthorizedError, 401, FailureKind.AUTH, RerankAuthError),
    (ForbiddenError, 403, FailureKind.AUTH, RerankAuthError),
    (InvalidTokenError, 498, FailureKind.AUTH, RerankAuthError),
    (NotFoundError, 404, FailureKind.NOT_FOUND, RerankNotFoundError),
)


def api_error(
    cls: type[ApiError],
    headers: dict[str, str] | None = None,
    body: object = "the provider said no",
) -> ApiError:
    """One of the SDK's errors; each concrete class fixes its own status."""
    return cls(body, headers or {})


@pytest.fixture
def handler() -> CohereErrorHandler:
    return CohereErrorHandler(retries=1, base_wait=0.0)


@pytest.mark.parametrize(("cls", "status", "kind", "domain_cls"), MAPPED)
def test_an_error_resolves_to_its_kind_and_class(
    handler: CohereErrorHandler,
    cls: type[ApiError],
    status: int,
    kind: FailureKind,
    domain_cls: type[Exception],
) -> None:
    error = api_error(cls)
    assert error.status_code == status
    assert handler.failure_kind(error) is kind
    assert handler.classify(error) == (domain_cls, kind in RETRYABLE_FAILURES)


def test_a_client_that_hung_up_is_retried(handler: CohereErrorHandler) -> None:
    error = api_error(ClientClosedRequestError)
    assert handler.classify(error) == (RerankConnectionError, True)


def test_an_unimplemented_endpoint_is_never_retried(
    handler: CohereErrorHandler,
) -> None:
    error = api_error(CohereNotImplementedError)
    assert handler.classify(error) == (RerankBadRequestError, False)


def test_a_rejected_token_is_never_retried(handler: CohereErrorHandler) -> None:
    assert handler.classify(api_error(InvalidTokenError)) == (
        RerankAuthError,
        False,
    )


def test_an_sdk_error_outside_the_map_is_unknown(
    handler: CohereErrorHandler,
) -> None:
    error = ApiError(status_code=418, headers={}, body="teapot")
    assert handler.failure_kind(error) is FailureKind.UNKNOWN
    assert handler.classify(error) == (RerankUnexpectedError, False)


def test_a_throttle_hint_is_read_off_the_headers(
    handler: CohereErrorHandler,
) -> None:
    error = api_error(TooManyRequestsError, headers={"retry-after": "4"})
    assert handler.retry_after(error) == 4.0


def test_a_throttle_hint_that_is_not_a_number_is_ignored(
    handler: CohereErrorHandler,
) -> None:
    error = api_error(TooManyRequestsError, headers={"retry-after": "later"})
    assert handler.retry_after(error) is None


def test_no_hint_is_read_from_anything_but_a_throttle(
    handler: CohereErrorHandler,
) -> None:
    error = api_error(InternalServerError, headers={"retry-after": "4"})
    assert handler.retry_after(error) is None


def test_no_hint_is_read_when_the_headers_are_empty(
    handler: CohereErrorHandler,
) -> None:
    assert handler.retry_after(api_error(TooManyRequestsError)) is None


@pytest.mark.parametrize("header", ["x-request-id", "request-id", "x-trace-id"])
def test_a_request_id_is_found_under_any_header_cohere_has_used(
    handler: CohereErrorHandler, header: str
) -> None:
    error = api_error(NotFoundError, headers={header: "trace-9"})
    wrapped = handler.wrap(error, RerankNotFoundError)
    assert (wrapped.status_code, wrapped.request_id) == (404, "trace-9")


def test_a_wrapped_error_without_headers_reports_no_request_id(
    handler: CohereErrorHandler,
) -> None:
    wrapped = handler.wrap(api_error(NotFoundError), RerankNotFoundError)
    assert wrapped.request_id is None


def test_an_exhausted_error_carries_the_attempts_and_the_status(
    handler: CohereErrorHandler,
) -> None:
    error = api_error(ServiceUnavailableError, headers={"x-request-id": "t-1"})
    wrapped = handler.wrap_exhausted(error, 4)
    assert isinstance(wrapped, RerankServerError)
    assert (wrapped.attempts, wrapped.status_code, wrapped.request_id) == (
        4,
        503,
        "t-1",
    )


def test_something_the_sdk_never_raised_becomes_the_unexpected_error(
    handler: CohereErrorHandler,
) -> None:
    error = ValueError("the reply did not parse")
    wrapped = handler.wrap_unexpected(error)
    assert isinstance(wrapped, RerankUnexpectedError)
    assert wrapped.provider_error is error


def test_every_sdk_exception_maps_to_a_sane_kind() -> None:
    """Walks the SDK's whole tree so a class it adds later is covered too."""
    handler = CohereErrorHandler()
    covered = 0
    for cls in subclasses_of(ApiError):
        error = build_sdk_error(cls)
        if error is None:
            continue
        covered += 1
        kind = handler.failure_kind(error)
        assert isinstance(kind, FailureKind)
        domain_cls, retryable = handler.classify(error)
        assert retryable is (kind in RETRYABLE_FAILURES)
        assert domain_cls is domain_class_for(handler, kind)
        assert issubclass(domain_cls, RerankError)
        if retryable:
            assert issubclass(domain_cls, ProviderRetryExhausted)
            assert handler.wrap_exhausted(error, 2).attempts == 2
        else:
            assert isinstance(handler.wrap(error, domain_cls), domain_cls)
    assert covered >= 11


def test_the_sdk_base_class_is_what_the_handler_catches() -> None:
    assert CohereErrorHandler().provider_error_type is ApiError
