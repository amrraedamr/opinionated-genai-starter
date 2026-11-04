"""OpenAI SDK errors, mapped onto the domain hierarchy."""

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
    raise MissingExtraError("llm", "openai", feature="OpenAIErrorHandler") from exc

from base.models.view_models.llm_exception import (
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
from base.services.llm.error_handler import AbstractErrorHandler

_ERROR_MAP: dict[type[OpenAIError], tuple[type[LLMError], bool]] = {
    # Retryable
    RateLimitError: (LLMRateLimitError, True),
    InternalServerError: (LLMServerError, True),
    APITimeoutError: (LLMTimeoutError, True),
    APIConnectionError: (LLMConnectionError, True),
    # Non-retryable
    BadRequestError: (LLMBadRequestError, False),
    UnprocessableEntityError: (LLMBadRequestError, False),
    AuthenticationError: (LLMAuthError, False),
    PermissionDeniedError: (LLMAuthError, False),
    NotFoundError: (LLMNotFoundError, False),
    ConflictError: (LLMConflictError, False),
}


class OpenAIErrorHandler(AbstractErrorHandler[OpenAIError]):
    """Classification for the OpenAI SDK. Retry policy lives in the base."""

    @property
    def provider_error_type(self) -> type[OpenAIError]:
        return OpenAIError

    def classify(self, error: OpenAIError) -> tuple[type[LLMError], bool]:
        """Resolve an OpenAI SDK error to `(domain class, is retryable)`.

        Walks the MRO so subclasses resolve like their parent. A
        `BadRequestError` carrying a content-filter marker is routed to
        `LLMContentFilterError`: Azure surfaces both prompt rejections and
        post-generation filters as HTTP 400 with `code == "content_filter"`.
        """
        if isinstance(error, BadRequestError) and self._is_content_filter(error):
            return LLMContentFilterError, False
        for cls in type(error).__mro__:
            if cls in _ERROR_MAP:
                return _ERROR_MAP[cls]
        return LLMUnexpectedError, False

    def identify(self, error: OpenAIError) -> tuple[int | None, str | None]:
        if isinstance(error, APIStatusError):
            return error.status_code, error.response.headers.get("x-request-id")
        if isinstance(error, APIError):
            return None, getattr(error, "request_id", None)
        return None, None

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

    def wrap(self, error: OpenAIError, domain_cls: type[LLMError]) -> LLMError:
        if domain_cls is LLMContentFilterError:
            status_code, request_id = self.identify(error)
            return LLMContentFilterError(
                getattr(error, "message", None) or str(error),
                incomplete_details=None,
                status_code=status_code,
                request_id=request_id,
                provider_error=error,
            )
        return super().wrap(error, domain_cls)

    @staticmethod
    def _is_content_filter(error: BadRequestError) -> bool:
        code = getattr(error, "code", None)
        if code == "content_filter":
            return True
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict) and inner.get("code") == "content_filter":
                return True
        message = getattr(error, "message", None) or str(error)
        return (
            "content management policy" in message.lower()
            or "content_filter" in message.lower()
        )
