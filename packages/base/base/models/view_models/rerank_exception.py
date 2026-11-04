"""Domain errors every rerank adapter raises. No provider SDK is named here.

Its own family, so a service that retrieves, embeds and reranks can tell which
of the three failed.
"""

from base.models.view_models.provider_exception import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class RerankError(ProviderError):
    """Base class for all rerank service errors."""


class RerankConfigError(RerankError, ProviderConfigError):
    """Invalid configuration (missing api key, unknown model, etc)."""


class RerankBadRequestError(RerankError):
    """400 / 422 — malformed request, too many documents, bad top_n."""


class RerankAuthError(RerankError):
    """401 / 403 — bad API key or insufficient permissions."""


class RerankNotFoundError(RerankError):
    """404 — model not found."""


class RerankConflictError(RerankError):
    """409 — conflict."""


class RerankRetryExhausted(RerankError, ProviderRetryExhausted):
    """Retries exhausted for a transient error."""


class RerankRateLimitError(RerankRetryExhausted):
    """429 after all retries exhausted."""


class RerankServerError(RerankRetryExhausted):
    """5xx after all retries exhausted."""


class RerankTimeoutError(RerankRetryExhausted):
    """Gateway or network timeout after all retries exhausted."""


class RerankConnectionError(RerankRetryExhausted):
    """Network connection failure after all retries exhausted."""


class RerankUnexpectedError(RerankError):
    """Anything we didn't anticipate. Always check `provider_error`."""
