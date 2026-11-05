"""Domain errors every blob adapter raises. No provider SDK is named here."""

from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class BlobError(ProviderError):
    """Base class for all blob storage errors."""


class BlobConfigError(BlobError, ProviderConfigError):
    """Invalid configuration (missing account url, bad credentials setup)."""


class BlobBadRequestError(BlobError):
    """400 / 422 — malformed request, bad container or blob name."""


class BlobAuthError(BlobError):
    """401 / 403 — bad credentials or insufficient permissions."""


class BlobNotFoundError(BlobError):
    """404 — container or blob does not exist."""


class BlobConflictError(BlobError):
    """409 / 412 — the blob already exists, or changed under us."""


class BlobRetryExhausted(BlobError, ProviderRetryExhausted):
    """Retries exhausted for a transient error."""


class BlobRateLimitError(BlobRetryExhausted):
    """429 after all retries exhausted. Azure throttles a busy account."""


class BlobServerError(BlobRetryExhausted):
    """5xx after all retries exhausted."""


class BlobTimeoutError(BlobRetryExhausted):
    """Network timeout after all retries exhausted."""


class BlobConnectionError(BlobRetryExhausted):
    """Network connection failure after all retries exhausted."""


class BlobUnexpectedError(BlobError):
    """Anything we didn't anticipate. Always check `provider_error`."""
