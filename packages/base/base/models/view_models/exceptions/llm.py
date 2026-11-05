"""Domain errors every LLM adapter raises. No provider SDK is named here.

Shape comes from `ProviderError`; what these add is the family, so catching an
LLM failure never catches an embedding or store failure.
"""

from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class LLMError(ProviderError):
    """Base class for all LLM service errors."""


class LLMConfigError(LLMError, ProviderConfigError):
    """Invalid configuration (missing model_name, bad api_host, etc)."""


class LLMBadRequestError(LLMError):
    """400 / 422 — malformed request, invalid params, schema violation."""


class LLMAuthError(LLMError):
    """401 / 403 — bad API key or insufficient permissions."""


class LLMNotFoundError(LLMError):
    """404 — model or resource not found."""


class LLMConflictError(LLMError):
    """409 — conflict."""


class LLMRetryExhausted(LLMError, ProviderRetryExhausted):
    """Retries exhausted for a transient error."""


class LLMRateLimitError(LLMRetryExhausted):
    """429 after all retries exhausted."""


class LLMServerError(LLMRetryExhausted):
    """5xx after all retries exhausted."""


class LLMTimeoutError(LLMRetryExhausted):
    """Network/API timeout after all retries exhausted."""


class LLMConnectionError(LLMRetryExhausted):
    """Network connection failure after all retries exhausted."""


class LLMStreamError(LLMError):
    """Mid-stream failure. Partial content may be available via `partial_content`."""

    def __init__(
        self,
        message: str,
        *,
        partial_content: str | None = None,
        provider_error: Exception | None = None,
    ) -> None:
        super().__init__(message, provider_error=provider_error)
        self.partial_content = partial_content


class LLMContentFilterError(LLMError):
    """Response blocked or truncated by the provider's content filter.

    Raised both for incomplete responses whose
    `incomplete_details.reason == "content_filter"` and for pre-flight HTTP 400
    responses whose body carries `code == "content_filter"`, which is how Azure
    OpenAI reports a prompt or response rejected upstream.
    """

    def __init__(
        self,
        message: str,
        *,
        incomplete_details: object | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        provider_error: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            request_id=request_id,
            provider_error=provider_error,
        )
        self.incomplete_details = incomplete_details


class LLMIncompleteError(LLMError):
    """Response incomplete for a reason other than content filter, e.g. max_tokens."""

    def __init__(self, message: str, *, incomplete_details: object) -> None:
        super().__init__(message)
        self.incomplete_details = incomplete_details


class LLMUnexpectedError(LLMError):
    """Anything we didn't anticipate. Always check `provider_error`."""
