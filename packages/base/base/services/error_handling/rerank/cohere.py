"""Cohere SDK errors, as the rerank family."""

from base.tools.errors import MissingExtraError

try:
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
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("rerank", "cohere", feature="CohereErrorHandler") from exc

from base.models.view_models.provider_exception import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.models.view_models.rerank_exception import (
    RerankAuthError,
    RerankBadRequestError,
    RerankConflictError,
    RerankConnectionError,
    RerankNotFoundError,
    RerankRateLimitError,
    RerankServerError,
    RerankTimeoutError,
    RerankUnexpectedError,
)
from base.services.error_handling import (
    RETRYABLE_FAILURES,
    AbstractErrorHandler,
    FailureKind,
)

_FAILURE_MAP: dict[type[ApiError], FailureKind] = {
    TooManyRequestsError: FailureKind.RATE_LIMIT,
    InternalServerError: FailureKind.SERVER,
    ServiceUnavailableError: FailureKind.SERVER,
    GatewayTimeoutError: FailureKind.TIMEOUT,
    ClientClosedRequestError: FailureKind.CONNECTION,
    BadRequestError: FailureKind.BAD_REQUEST,
    UnprocessableEntityError: FailureKind.BAD_REQUEST,
    CohereNotImplementedError: FailureKind.BAD_REQUEST,
    UnauthorizedError: FailureKind.AUTH,
    ForbiddenError: FailureKind.AUTH,
    InvalidTokenError: FailureKind.AUTH,
    NotFoundError: FailureKind.NOT_FOUND,
}

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: RerankRateLimitError,
    FailureKind.SERVER: RerankServerError,
    FailureKind.TIMEOUT: RerankTimeoutError,
    FailureKind.CONNECTION: RerankConnectionError,
    FailureKind.BAD_REQUEST: RerankBadRequestError,
    FailureKind.AUTH: RerankAuthError,
    FailureKind.NOT_FOUND: RerankNotFoundError,
    FailureKind.CONFLICT: RerankConflictError,
    FailureKind.UNKNOWN: RerankUnexpectedError,
}

_REQUEST_ID_HEADERS: tuple[str, ...] = ("x-request-id", "request-id", "x-trace-id")
"""Cohere has moved the header between versions, so try the ones it has used."""


class CohereErrorHandler(AbstractErrorHandler[ApiError]):
    """Cohere rerank calls, raising the rerank family.

    Cohere serves only reranking here, so SDK reading and domain mapping sit
    together. Were it ever to serve a second capability, the SDK half would
    move up beside `error_handling/openai.py`.
    """

    @property
    def provider_error_type(self) -> type[ApiError]:
        return ApiError

    def failure_kind(self, error: ApiError) -> FailureKind:
        """Walks the MRO so a subclass resolves like its parent."""
        for cls in type(error).__mro__:
            if cls in _FAILURE_MAP:
                return _FAILURE_MAP[cls]
        return FailureKind.UNKNOWN

    def classify(self, error: ApiError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return _ERROR_MAP[kind], kind in RETRYABLE_FAILURES

    def retry_after(self, error: ApiError) -> float | None:
        """Cohere sends `retry-after` on a 429; honour it over backoff."""
        if not isinstance(error, TooManyRequestsError):
            return None
        headers = getattr(error, "headers", None) or {}
        header = headers.get("retry-after")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None

    def wrap(self, error: ApiError, domain_cls: type[Exception]) -> Exception:
        status_code, request_id = self._identify(error)
        return domain_cls(
            str(error),
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    def wrap_exhausted(self, error: ApiError, attempts: int) -> Exception:
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
        return RerankUnexpectedError.from_provider(error)

    @staticmethod
    def _identify(error: ApiError) -> tuple[int | None, str | None]:
        """`(status code, request id)` from an SDK error."""
        headers = getattr(error, "headers", None) or {}
        request_id = next(
            (headers[name] for name in _REQUEST_ID_HEADERS if name in headers), None
        )
        return getattr(error, "status_code", None), request_id
