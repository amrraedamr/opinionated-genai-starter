"""Anthropic SDK errors, mapped onto the domain hierarchy."""

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

from base.models.view_models.llm_exception import (
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
from base.services.llm.error_handler import AbstractErrorHandler

_ERROR_MAP: dict[type[AnthropicError], tuple[type[LLMError], bool]] = {
    # Retryable
    RateLimitError: (LLMRateLimitError, True),
    InternalServerError: (LLMServerError, True),
    OverloadedError: (LLMServerError, True),
    ServiceUnavailableError: (LLMServerError, True),
    APITimeoutError: (LLMTimeoutError, True),
    DeadlineExceededError: (LLMTimeoutError, True),
    APIConnectionError: (LLMConnectionError, True),
    # Non-retryable
    BadRequestError: (LLMBadRequestError, False),
    UnprocessableEntityError: (LLMBadRequestError, False),
    RequestTooLargeError: (LLMBadRequestError, False),
    AuthenticationError: (LLMAuthError, False),
    PermissionDeniedError: (LLMAuthError, False),
    CredentialsError: (LLMAuthError, False),
    NotFoundError: (LLMNotFoundError, False),
    ConflictError: (LLMConflictError, False),
}


class AnthropicErrorHandler(AbstractErrorHandler[AnthropicError]):
    """Classification for the Anthropic SDK. Retry policy lives in the base."""

    @property
    def provider_error_type(self) -> type[AnthropicError]:
        return AnthropicError

    def classify(self, error: AnthropicError) -> tuple[type[LLMError], bool]:
        """Resolve an Anthropic SDK error to `(domain class, is retryable)`.

        Walks the MRO so subclasses resolve like their parent. There is no
        content-filter arm: Anthropic reports a refusal as a `stop_reason` on a
        successful response, so the adapter raises that, not this.
        """
        for cls in type(error).__mro__:
            if cls in _ERROR_MAP:
                return _ERROR_MAP[cls]
        return LLMUnexpectedError, False

    def identify(self, error: AnthropicError) -> tuple[int | None, str | None]:
        if isinstance(error, APIStatusError):
            return error.status_code, error.response.headers.get("request-id")
        if isinstance(error, APIError):
            return None, getattr(error, "request_id", None)
        return None, None

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
