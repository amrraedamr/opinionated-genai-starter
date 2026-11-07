"""How the Azure SDK's exceptions are read, and the four domains that share it.

Azure has no exception type for throttling or a failed server — both arrive as
a plain `HttpResponseError` — so the status code does that work, and the
boundaries are what this file pins. The blob, Key Vault, Cosmos and AI Search
handlers differ only in which family they raise, so they are checked for
agreeing on the kind and disagreeing on nothing else.
"""

from __future__ import annotations

import pytest
from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceRequestTimeoutError,
    ServiceResponseError,
    ServiceResponseTimeoutError,
)

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.blob import (
    BlobError,
    BlobNotFoundError,
    BlobRateLimitError,
    BlobServerError,
    BlobUnexpectedError,
)
from base.models.view_models.exceptions.keyvault import KeyVaultError
from base.models.view_models.exceptions.nosql import NoSqlError, NoSqlRateLimitError
from base.models.view_models.exceptions.provider import ProviderRetryExhausted
from base.models.view_models.exceptions.vector_search import VectorSearchError
from base.services.error_handling.blob.azure import AzureBlobErrorHandler
from base.services.error_handling.keyvault.azure import AzureKeyVaultErrorHandler
from base.services.error_handling.nosql.cosmos import CosmosErrorHandler
from base.services.error_handling.vector_search.azure import AzureSearchErrorHandler
from tests.fakes import subclasses_of
from tests.unit.error_handling.support import (
    AzureResponse,
    build_sdk_error,
    domain_class_for,
)

HANDLERS: tuple[tuple[type, type[Exception]], ...] = (
    (AzureBlobErrorHandler, BlobError),
    (AzureKeyVaultErrorHandler, KeyVaultError),
    (CosmosErrorHandler, NoSqlError),
    (AzureSearchErrorHandler, VectorSearchError),
)

STATUS_KINDS: tuple[tuple[int, FailureKind], ...] = (
    (429, FailureKind.RATE_LIMIT),
    (500, FailureKind.SERVER),
    (503, FailureKind.SERVER),
    (401, FailureKind.AUTH),
    (403, FailureKind.AUTH),
    (404, FailureKind.NOT_FOUND),
    (409, FailureKind.CONFLICT),
    (412, FailureKind.CONFLICT),
    (400, FailureKind.BAD_REQUEST),
    (422, FailureKind.BAD_REQUEST),
)


def http_error(
    status: int, headers: dict[str, str] | None = None, message: str = "azure said no"
) -> HttpResponseError:
    """The one class Azure reports most failures as, carrying a status."""
    return HttpResponseError(message=message, response=AzureResponse(status, headers))


@pytest.fixture
def handler() -> AzureBlobErrorHandler:
    return AzureBlobErrorHandler(retries=1, base_wait=0.0)


@pytest.mark.parametrize(("status", "kind"), STATUS_KINDS)
def test_a_status_resolves_to_its_kind(
    handler: AzureBlobErrorHandler, status: int, kind: FailureKind
) -> None:
    assert handler.failure_kind(http_error(status)) is kind


@pytest.mark.parametrize(("status", "kind"), STATUS_KINDS)
def test_retryability_follows_the_kind_and_nothing_else(
    handler: AzureBlobErrorHandler, status: int, kind: FailureKind
) -> None:
    _, retryable = handler.classify(http_error(status))
    assert retryable is (kind in RETRYABLE_FAILURES)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ServiceRequestTimeoutError("slow"), FailureKind.TIMEOUT),
        (ServiceResponseTimeoutError("slow"), FailureKind.TIMEOUT),
        (ServiceRequestError("unreachable"), FailureKind.CONNECTION),
        (ServiceResponseError("truncated"), FailureKind.CONNECTION),
        (ClientAuthenticationError("bad token"), FailureKind.AUTH),
        (ResourceNotFoundError("gone"), FailureKind.NOT_FOUND),
        (ResourceExistsError("already there"), FailureKind.CONFLICT),
        (ResourceModifiedError("etag moved"), FailureKind.CONFLICT),
    ],
)
def test_an_error_the_sdk_named_resolves_on_its_type(
    handler: AzureBlobErrorHandler, error: AzureError, kind: FailureKind
) -> None:
    assert handler.failure_kind(error) is kind


def test_the_exception_type_is_read_before_the_status(
    handler: AzureBlobErrorHandler,
) -> None:
    """`ResourceNotFoundError` is an `HttpResponseError`; its type must win."""
    error = ResourceNotFoundError(message="gone", response=AzureResponse(500))
    assert error.status_code == 500
    assert handler.failure_kind(error) is FailureKind.NOT_FOUND


def test_a_throttled_request_is_retried(handler: AzureBlobErrorHandler) -> None:
    assert handler.classify(http_error(429)) == (BlobRateLimitError, True)


def test_a_failed_server_is_retried(handler: AzureBlobErrorHandler) -> None:
    assert handler.classify(http_error(503)) == (BlobServerError, True)


def test_a_missing_blob_is_never_retried(handler: AzureBlobErrorHandler) -> None:
    assert handler.classify(http_error(404)) == (BlobNotFoundError, False)


def test_an_error_with_no_status_at_all_is_unknown(
    handler: AzureBlobErrorHandler,
) -> None:
    error = HttpResponseError(message="no response at all")
    assert error.status_code is None
    assert handler.failure_kind(error) is FailureKind.UNKNOWN


def test_a_status_below_four_hundred_is_unknown(
    handler: AzureBlobErrorHandler,
) -> None:
    assert handler.failure_kind(http_error(304)) is FailureKind.UNKNOWN


def test_an_error_outside_the_map_is_unknown(
    handler: AzureBlobErrorHandler,
) -> None:
    error = AzureError("something new")
    assert handler.failure_kind(error) is FailureKind.UNKNOWN
    assert handler.classify(error) == (BlobUnexpectedError, False)


@pytest.mark.parametrize("header", ["Retry-After", "retry-after"])
def test_a_throttle_hint_is_read_whichever_case_it_arrives_in(
    handler: AzureBlobErrorHandler, header: str
) -> None:
    assert handler.retry_after(http_error(429, {header: "8"})) == 8.0


def test_a_throttle_hint_that_is_not_a_number_is_ignored(
    handler: AzureBlobErrorHandler,
) -> None:
    assert handler.retry_after(http_error(429, {"Retry-After": "soon"})) is None


def test_no_hint_is_read_when_the_header_is_absent(
    handler: AzureBlobErrorHandler,
) -> None:
    assert handler.retry_after(http_error(429)) is None


def test_a_hint_is_read_from_any_status_because_azure_sends_it_widely(
    handler: AzureBlobErrorHandler,
) -> None:
    assert handler.retry_after(http_error(503, {"Retry-After": "2"})) == 2.0


@pytest.mark.parametrize("header", ["x-ms-request-id", "x-ms-client-request-id"])
def test_a_request_id_is_found_under_either_azure_header(
    handler: AzureBlobErrorHandler, header: str
) -> None:
    wrapped = handler.wrap(http_error(404, {header: "op-5"}), BlobNotFoundError)
    assert (wrapped.status_code, wrapped.request_id) == (404, "op-5")


def test_the_server_request_id_is_preferred_over_the_client_one(
    handler: AzureBlobErrorHandler,
) -> None:
    headers = {"x-ms-client-request-id": "mine", "x-ms-request-id": "theirs"}
    wrapped = handler.wrap(http_error(404, headers), BlobNotFoundError)
    assert wrapped.request_id == "theirs"


def test_a_wrapped_error_without_headers_reports_no_request_id(
    handler: AzureBlobErrorHandler,
) -> None:
    wrapped = handler.wrap(http_error(404), BlobNotFoundError)
    assert wrapped.request_id is None


def test_an_exhausted_error_carries_the_attempts_and_the_status(
    handler: AzureBlobErrorHandler,
) -> None:
    error = http_error(503, {"x-ms-request-id": "op-9"})
    wrapped = handler.wrap_exhausted(error, 4)
    assert isinstance(wrapped, BlobServerError)
    assert (wrapped.attempts, wrapped.status_code, wrapped.request_id) == (
        4,
        503,
        "op-9",
    )


def test_something_the_sdk_never_raised_becomes_the_unexpected_error(
    handler: AzureBlobErrorHandler,
) -> None:
    error = ValueError("the payload did not parse")
    wrapped = handler.wrap_unexpected(error)
    assert isinstance(wrapped, BlobUnexpectedError)
    assert wrapped.provider_error is error


def test_cosmos_reads_the_same_statuses_but_raises_the_store_family() -> None:
    assert CosmosErrorHandler().classify(http_error(429)) == (
        NoSqlRateLimitError,
        True,
    )


def test_the_four_azure_handlers_agree_on_every_kind() -> None:
    instances = [handler_cls() for handler_cls, _ in HANDLERS]
    for status, kind in STATUS_KINDS:
        error = http_error(status)
        assert {instance.failure_kind(error) for instance in instances} == {kind}


def test_the_four_azure_handlers_never_share_a_domain_class() -> None:
    classes = [
        domain_class_for(handler_cls(), FailureKind.NOT_FOUND)
        for handler_cls, _ in HANDLERS
    ]
    assert len(set(classes)) == len(HANDLERS)


@pytest.mark.parametrize(("handler_cls", "family"), HANDLERS)
def test_every_sdk_exception_maps_to_a_sane_kind(
    handler_cls: type, family: type[Exception]
) -> None:
    """Walks the SDK's whole tree so a class it adds later is covered too."""
    handler = handler_cls()
    covered = 0
    for cls in subclasses_of(AzureError):
        error = build_sdk_error(cls)
        if error is None:
            continue
        covered += 1
        kind = handler.failure_kind(error)
        assert isinstance(kind, FailureKind)
        domain_cls, retryable = handler.classify(error)
        assert retryable is (kind in RETRYABLE_FAILURES)
        assert domain_cls is domain_class_for(handler, kind)
        assert issubclass(domain_cls, family)
        if retryable:
            assert issubclass(domain_cls, ProviderRetryExhausted)
            assert handler.wrap_exhausted(error, 2).attempts == 2
        else:
            assert isinstance(handler.wrap(error, domain_cls), domain_cls)
    assert covered >= 12


@pytest.mark.parametrize(("handler_cls", "family"), HANDLERS)
def test_the_sdk_base_class_is_what_every_azure_handler_catches(
    handler_cls: type, family: type[Exception]
) -> None:
    del family
    assert handler_cls().provider_error_type is AzureError
