from base.tools.errors import MissingExtraError

try:
    from openai import APIError, APIStatusError
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="LLM error models") from exc


class LLMError(Exception):
    """Base class for all LLM service errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        provider_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.provider_error = provider_error

    @classmethod
    def from_openai(
        cls,
        error: Exception,
        message: str | None = None,
    ) -> "LLMError":
        """Wrap an OpenAI SDK exception, preserving status code and request id."""
        status_code: int | None = None
        request_id: str | None = None

        if isinstance(error, APIStatusError):
            status_code = error.status_code
            request_id = error.response.headers.get("x-request-id")
        elif isinstance(error, APIError):
            request_id = getattr(error, "request_id", None)

        return cls(
            message or str(error),
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    def __str__(self) -> str:
        parts: list[str] = [super().__str__()]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.request_id is not None:
            parts.append(f"request_id={self.request_id}")
        return " | ".join(parts)


class LLMConfigError(LLMError):
    """Invalid configuration (missing model_name, bad api_host, etc)."""


class LLMBadRequestError(LLMError):
    """400 / 422 — malformed request, invalid params, schema violation."""


class LLMAuthError(LLMError):
    """401 / 403 — bad API key or insufficient permissions."""


class LLMNotFoundError(LLMError):
    """404 — model or resource not found."""


class LLMConflictError(LLMError):
    """409 — conflict."""


class LLMRetryExhausted(LLMError):
    """Retries exhausted for a transient error. Inspect .provider_error for the cause."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
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
        self.attempts = attempts

    def __str__(self) -> str:
        return f"{super().__str__()} | attempts={self.attempts}"


class LLMRateLimitError(LLMRetryExhausted):
    """429 after all retries exhausted."""


class LLMServerError(LLMRetryExhausted):
    """5xx after all retries exhausted."""


class LLMTimeoutError(LLMRetryExhausted):
    """Network/API timeout after all retries exhausted."""


class LLMConnectionError(LLMRetryExhausted):
    """Network connection failure after all retries exhausted."""


class LLMStreamError(LLMError):
    """Mid-stream failure. Partial content may be available via .partial_content."""

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

    Triggered both for streaming/incomplete responses with
    ``incomplete_details.reason == "content_filter"`` AND for pre-flight
    HTTP 400 responses where the body's ``code == "content_filter"``
    (typical of Azure OpenAI when the content filter rejects the prompt
    or response upstream).
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
    """Response incomplete for reason other than content filter (e.g., max_tokens)."""

    def __init__(self, message: str, *, incomplete_details: object) -> None:
        super().__init__(message)
        self.incomplete_details = incomplete_details


class LLMUnexpectedError(LLMError):
    """Anything we didn't anticipate. Always check .provider_error."""
