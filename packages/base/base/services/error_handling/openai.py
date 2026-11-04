"""Reading the OpenAI SDK's exceptions, without deciding what callers see.

Which kind of failure an exception represents is a property of the SDK, so it
is stated once here. Which error class a kind becomes belongs to the domain
making the call, so subclasses supply that — the LLM adapters and the embedding
adapter share everything below.
"""

from abc import ABC, abstractmethod

from base.tools.errors import MissingExtraError

try:
    from openai import (
        APIConnectionError,
        APIError,
        APIStatusError,
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
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="OpenAI error handling") from exc

from base.models.view_models.provider_exception import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.services.error_handling import (
    RETRYABLE_FAILURES,
    AbstractErrorHandler,
    FailureKind,
)

_FAILURE_MAP: dict[type[OpenAIError], FailureKind] = {
    RateLimitError: FailureKind.RATE_LIMIT,
    InternalServerError: FailureKind.SERVER,
    APITimeoutError: FailureKind.TIMEOUT,
    APIConnectionError: FailureKind.CONNECTION,
    BadRequestError: FailureKind.BAD_REQUEST,
    UnprocessableEntityError: FailureKind.BAD_REQUEST,
    AuthenticationError: FailureKind.AUTH,
    PermissionDeniedError: FailureKind.AUTH,
    NotFoundError: FailureKind.NOT_FOUND,
    ConflictError: FailureKind.CONFLICT,
}


class OpenAISDKErrorHandler(AbstractErrorHandler[OpenAIError], ABC):
    """The OpenAI SDK half of error handling. Subclasses choose the domain."""

    @abstractmethod
    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        """The domain error class this kind of failure becomes."""

    @property
    def provider_error_type(self) -> type[OpenAIError]:
        return OpenAIError

    def failure_kind(self, error: OpenAIError) -> FailureKind:
        """Which kind of failure an SDK exception is.

        Walks the MRO so a subclass resolves like its parent.
        """
        for cls in type(error).__mro__:
            if cls in _FAILURE_MAP:
                return _FAILURE_MAP[cls]
        return FailureKind.UNKNOWN

    def classify(self, error: OpenAIError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return self.error_for(kind), kind in RETRYABLE_FAILURES

    def retry_after(self, error: OpenAIError) -> float | None:
        """OpenAI sends `Retry-After` on a 429; honour it over backoff."""
        if not isinstance(error, RateLimitError):
            return None
        response = getattr(error, "response", None)
        if response is None:
            return None
        header = response.headers.get("retry-after")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None

    def wrap(self, error: OpenAIError, domain_cls: type[Exception]) -> Exception:
        status_code, request_id = self._identify(error)
        return domain_cls(
            str(error),
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    def wrap_exhausted(self, error: OpenAIError, attempts: int) -> Exception:
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
    def _identify(error: OpenAIError) -> tuple[int | None, str | None]:
        """`(status code, request id)` from an SDK error."""
        if isinstance(error, APIStatusError):
            return error.status_code, error.response.headers.get("x-request-id")
        if isinstance(error, APIError):
            return None, getattr(error, "request_id", None)
        return None, None
