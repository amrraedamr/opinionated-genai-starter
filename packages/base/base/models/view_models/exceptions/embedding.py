"""Domain errors every embedding adapter raises. No provider SDK is named here.

Separate from the LLM family on purpose: a service that embeds and generates
should be able to catch a failure of one without catching the other.
"""

from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class EmbeddingError(ProviderError):
    """Base class for all embedding service errors."""


class EmbeddingConfigError(EmbeddingError, ProviderConfigError):
    """Invalid configuration (missing model_name, bad api_host, etc)."""


class EmbeddingBadRequestError(EmbeddingError):
    """400 / 422 — malformed request, input too long, bad dimensions."""


class EmbeddingAuthError(EmbeddingError):
    """401 / 403 — bad API key or insufficient permissions."""


class EmbeddingNotFoundError(EmbeddingError):
    """404 — model or deployment not found."""


class EmbeddingConflictError(EmbeddingError):
    """409 — conflict."""


class EmbeddingRetryExhausted(EmbeddingError, ProviderRetryExhausted):
    """Retries exhausted for a transient error."""


class EmbeddingRateLimitError(EmbeddingRetryExhausted):
    """429 after all retries exhausted."""


class EmbeddingServerError(EmbeddingRetryExhausted):
    """5xx after all retries exhausted."""


class EmbeddingTimeoutError(EmbeddingRetryExhausted):
    """Network/API timeout after all retries exhausted."""


class EmbeddingConnectionError(EmbeddingRetryExhausted):
    """Network connection failure after all retries exhausted."""


class EmbeddingUnexpectedError(EmbeddingError):
    """Anything we didn't anticipate. Always check `provider_error`."""
