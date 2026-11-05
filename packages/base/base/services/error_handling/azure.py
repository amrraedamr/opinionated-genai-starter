"""Reading the Azure SDK's exceptions, without deciding what callers see.

Shared by every Azure-backed service. Which kind of failure an exception is, is
a property of the SDK, so it is stated once; which error class a kind becomes
belongs to the domain making the call, so subclasses supply that.
"""

from abc import ABC, abstractmethod
from http import HTTPStatus

from base.tools.errors import MissingExtraError

try:
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
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "azure", "azure-core", feature="Azure error handling"
    ) from exc

from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.services.error_handling import (
    RETRYABLE_FAILURES,
    AbstractErrorHandler,
    FailureKind,
)

_FAILURE_MAP: dict[type[AzureError], FailureKind] = {
    ServiceRequestTimeoutError: FailureKind.TIMEOUT,
    ServiceResponseTimeoutError: FailureKind.TIMEOUT,
    ServiceRequestError: FailureKind.CONNECTION,
    ServiceResponseError: FailureKind.CONNECTION,
    ClientAuthenticationError: FailureKind.AUTH,
    ResourceNotFoundError: FailureKind.NOT_FOUND,
    ResourceExistsError: FailureKind.CONFLICT,
    ResourceModifiedError: FailureKind.CONFLICT,
}

_REQUEST_ID_HEADERS: tuple[str, ...] = ("x-ms-request-id", "x-ms-client-request-id")


class AzureSDKErrorHandler(AbstractErrorHandler[AzureError], ABC):
    """The Azure SDK half of error handling. Subclasses choose the domain.

    Azure has no exception type for throttling or a failed server, reporting
    both as a plain `HttpResponseError`, so those are read off the status code.
    """

    @abstractmethod
    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        """The domain error class this kind of failure becomes."""

    @property
    def provider_error_type(self) -> type[AzureError]:
        return AzureError

    def failure_kind(self, error: AzureError) -> FailureKind:
        """Walks the MRO so a subclass resolves like its parent."""
        for cls in type(error).__mro__:
            if cls in _FAILURE_MAP:
                return _FAILURE_MAP[cls]
        if isinstance(error, HttpResponseError):
            return self._kind_for_status(error.status_code)
        return FailureKind.UNKNOWN

    def classify(self, error: AzureError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return self.error_for(kind), kind in RETRYABLE_FAILURES

    def retry_after(self, error: AzureError) -> float | None:
        """Azure sends `Retry-After` when it throttles; honour it over backoff."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) or {}
        header = headers.get("Retry-After") or headers.get("retry-after")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None

    def wrap(self, error: AzureError, domain_cls: type[Exception]) -> Exception:
        status_code, request_id = self._identify(error)
        return domain_cls(
            str(error),
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    def wrap_exhausted(self, error: AzureError, attempts: int) -> Exception:
        domain_cls, _ = self.classify(error)
        assert issubclass(domain_cls, ProviderRetryExhausted), (
            f"Retryable error {type(error).__name__} mapped to non-retry-exhausted "
            f"domain class {domain_cls.__name__}"
        )
        status_code, request_id = self._identify(error)
        return domain_cls(
            f"{domain_cls.__name__} after {attempts} attempts: {error}",
            attempts=attempts,
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    def wrap_unexpected(self, error: Exception) -> Exception:
        return self.error_for(FailureKind.UNKNOWN).from_provider(error)

    @staticmethod
    def _kind_for_status(status: int | None) -> FailureKind:
        if status is None:
            return FailureKind.UNKNOWN
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            return FailureKind.RATE_LIMIT
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
            return FailureKind.SERVER
        if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
            return FailureKind.AUTH
        if status == HTTPStatus.NOT_FOUND:
            return FailureKind.NOT_FOUND
        if status in (HTTPStatus.CONFLICT, HTTPStatus.PRECONDITION_FAILED):
            return FailureKind.CONFLICT
        if status >= HTTPStatus.BAD_REQUEST:
            return FailureKind.BAD_REQUEST
        return FailureKind.UNKNOWN

    @staticmethod
    def _identify(error: AzureError) -> tuple[int | None, str | None]:
        """`(status code, request id)` from an SDK error."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) or {}
        request_id = next(
            (headers[name] for name in _REQUEST_ID_HEADERS if name in headers), None
        )
        return getattr(error, "status_code", None), request_id
