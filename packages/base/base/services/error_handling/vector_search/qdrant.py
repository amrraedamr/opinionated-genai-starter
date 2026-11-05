"""Qdrant SDK errors, as the vector search family."""

from http import HTTPStatus

from base.tools.errors import MissingExtraError

try:
    from qdrant_client.http.exceptions import (
        ApiException,
        ResponseHandlingException,
        UnexpectedResponse,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "vectors", "qdrant-client", feature="QdrantErrorHandler"
    ) from exc

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.models.view_models.exceptions.vector_search import (
    VectorSearchAuthError,
    VectorSearchBadRequestError,
    VectorSearchConflictError,
    VectorSearchConnectionError,
    VectorSearchNotFoundError,
    VectorSearchRateLimitError,
    VectorSearchServerError,
    VectorSearchTimeoutError,
    VectorSearchUnexpectedError,
)
from base.services.error_handling import AbstractErrorHandler

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: VectorSearchRateLimitError,
    FailureKind.SERVER: VectorSearchServerError,
    FailureKind.TIMEOUT: VectorSearchTimeoutError,
    FailureKind.CONNECTION: VectorSearchConnectionError,
    FailureKind.BAD_REQUEST: VectorSearchBadRequestError,
    FailureKind.AUTH: VectorSearchAuthError,
    FailureKind.NOT_FOUND: VectorSearchNotFoundError,
    FailureKind.CONFLICT: VectorSearchConflictError,
    FailureKind.UNKNOWN: VectorSearchUnexpectedError,
}


class QdrantErrorHandler(AbstractErrorHandler[ApiException]):
    """Qdrant calls, raising the vector search family.

    Qdrant has two exception shapes rather than a type per failure: a response
    it did not expect, which carries the status code, and a transport problem
    reaching the server at all.
    """

    @property
    def provider_error_type(self) -> type[ApiException]:
        return ApiException

    def failure_kind(self, error: ApiException) -> FailureKind:
        if isinstance(error, ResponseHandlingException):
            return FailureKind.CONNECTION
        if isinstance(error, UnexpectedResponse):
            return self._kind_for_status(error.status_code)
        return FailureKind.UNKNOWN

    def classify(self, error: ApiException) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return _ERROR_MAP[kind], kind in RETRYABLE_FAILURES

    def retry_after(self, error: ApiException) -> float | None:
        """Qdrant sends `retry-after` when it sheds load; honour it over backoff."""
        headers = getattr(error, "headers", None) or {}
        header = headers.get("retry-after") or headers.get("Retry-After")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None

    def wrap(self, error: ApiException, domain_cls: type[Exception]) -> Exception:
        return domain_cls(
            str(error),
            status_code=self._status(error),
            provider_error=error,
        )

    def wrap_exhausted(self, error: ApiException, attempts: int) -> Exception:
        domain_cls, _ = self.classify(error)
        assert issubclass(domain_cls, ProviderRetryExhausted), (
            f"Retryable error {type(error).__name__} mapped to non-retry-exhausted "
            f"domain class {domain_cls.__name__}"
        )
        return domain_cls(
            f"{domain_cls.__name__} after {attempts} attempts: {error}",
            attempts=attempts,
            status_code=self._status(error),
            provider_error=error,
        )

    def wrap_unexpected(self, error: Exception) -> Exception:
        return VectorSearchUnexpectedError.from_provider(error)

    @staticmethod
    def _status(error: ApiException) -> int | None:
        return getattr(error, "status_code", None)

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
        if status == HTTPStatus.CONFLICT:
            return FailureKind.CONFLICT
        if status >= HTTPStatus.BAD_REQUEST:
            return FailureKind.BAD_REQUEST
        return FailureKind.UNKNOWN
