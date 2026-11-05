"""Domain errors every vector search adapter raises. No provider SDK is named here."""

from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class VectorSearchError(ProviderError):
    """Base class for all vector search errors."""


class VectorSearchConfigError(VectorSearchError, ProviderConfigError):
    """Invalid configuration (missing url, bad api key, unknown distance)."""


class VectorSearchBadRequestError(VectorSearchError):
    """400 / 422 — wrong vector width, malformed filter, bad payload."""


class VectorSearchAuthError(VectorSearchError):
    """401 / 403 — bad api key or insufficient permissions."""


class VectorSearchNotFoundError(VectorSearchError):
    """404 — the collection or point does not exist."""


class VectorSearchConflictError(VectorSearchError):
    """409 — the collection already exists, or changed under us."""


class VectorSearchRetryExhausted(VectorSearchError, ProviderRetryExhausted):
    """Retries exhausted for a transient error."""


class VectorSearchRateLimitError(VectorSearchRetryExhausted):
    """429 after all retries exhausted."""


class VectorSearchServerError(VectorSearchRetryExhausted):
    """5xx after all retries exhausted."""


class VectorSearchTimeoutError(VectorSearchRetryExhausted):
    """Network timeout after all retries exhausted."""


class VectorSearchConnectionError(VectorSearchRetryExhausted):
    """Network connection failure after all retries exhausted."""


class VectorSearchUnexpectedError(VectorSearchError):
    """Anything we didn't anticipate. Always check `provider_error`."""
