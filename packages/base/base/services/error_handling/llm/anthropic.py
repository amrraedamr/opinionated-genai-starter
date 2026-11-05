"""Anthropic SDK errors, as the LLM family."""

from base.tools.errors import MissingExtraError

try:
    from anthropic import (
        AnthropicError,
        APIConnectionError,
        APIError,
        APIStatusError,
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
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "anthropic", "anthropic", feature="AnthropicErrorHandler"
    ) from exc

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.llm import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConflictError,
    LLMConnectionError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnexpectedError,
)
from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.services.error_handling import AbstractErrorHandler

_FAILURE_MAP: dict[type[AnthropicError], FailureKind] = {
    RateLimitError: FailureKind.RATE_LIMIT,
    InternalServerError: FailureKind.SERVER,
    OverloadedError: FailureKind.SERVER,
    ServiceUnavailableError: FailureKind.SERVER,
    APITimeoutError: FailureKind.TIMEOUT,
    DeadlineExceededError: FailureKind.TIMEOUT,
    APIConnectionError: FailureKind.CONNECTION,
    BadRequestError: FailureKind.BAD_REQUEST,
    UnprocessableEntityError: FailureKind.BAD_REQUEST,
    RequestTooLargeError: FailureKind.BAD_REQUEST,
    AuthenticationError: FailureKind.AUTH,
    PermissionDeniedError: FailureKind.AUTH,
    CredentialsError: FailureKind.AUTH,
    NotFoundError: FailureKind.NOT_FOUND,
    ConflictError: FailureKind.CONFLICT,
}

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: LLMRateLimitError,
    FailureKind.SERVER: LLMServerError,
    FailureKind.TIMEOUT: LLMTimeoutError,
    FailureKind.CONNECTION: LLMConnectionError,
    FailureKind.BAD_REQUEST: LLMBadRequestError,
    FailureKind.AUTH: LLMAuthError,
    FailureKind.NOT_FOUND: LLMNotFoundError,
    FailureKind.CONFLICT: LLMConflictError,
    FailureKind.UNKNOWN: LLMUnexpectedError,
}


class AnthropicErrorHandler(AbstractErrorHandler[AnthropicError]):
    """Anthropic generation calls, raising the LLM family.

    Anthropic serves only generation here, so SDK reading and domain mapping
    sit together. Were it ever to serve a second capability, the SDK half would
    move up beside `error_handling/openai.py`.

    There is no content-filter arm: Anthropic reports a refusal as a
    `stop_reason` on a successful response, so the adapter raises that, not this.
    """

    @property
    def provider_error_type(self) -> type[AnthropicError]:
        return AnthropicError

    def failure_kind(self, error: AnthropicError) -> FailureKind:
        """Walks the MRO so a subclass resolves like its parent."""
        for cls in type(error).__mro__:
            if cls in _FAILURE_MAP:
                return _FAILURE_MAP[cls]
        return FailureKind.UNKNOWN

    def classify(self, error: AnthropicError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return _ERROR_MAP[kind], kind in RETRYABLE_FAILURES

    def retry_after(self, error: AnthropicError) -> float | None:
        """Anthropic sends `retry-after` on a 429; honour it over backoff."""
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

    def wrap(self, error: AnthropicError, domain_cls: type[Exception]) -> Exception:
        status_code, request_id = self._identify(error)
        return domain_cls(
            str(error),
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    def wrap_exhausted(self, error: AnthropicError, attempts: int) -> Exception:
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
        return LLMUnexpectedError.from_provider(error)

    @staticmethod
    def _identify(error: AnthropicError) -> tuple[int | None, str | None]:
        """`(status code, request id)` from an SDK error."""
        if isinstance(error, APIStatusError):
            return error.status_code, error.response.headers.get("request-id")
        if isinstance(error, APIError):
            return None, getattr(error, "request_id", None)
        return None, None
