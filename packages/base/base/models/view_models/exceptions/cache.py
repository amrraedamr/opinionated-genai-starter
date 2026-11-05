"""Domain errors every cache adapter raises. No provider SDK is named here."""

from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class CacheError(ProviderError):
    """Base class for all cache errors."""


class CacheConfigError(CacheError, ProviderConfigError):
    """Invalid configuration. Trying again will not help."""


class CacheBadRequestError(CacheError):
    """Malformed command or value the server refused."""


class CacheAuthError(CacheError):
    """Bad credentials or insufficient permissions."""


class CacheNotFoundError(CacheError):
    """The key or script does not exist."""


class CacheConflictError(CacheError):
    """A watched key changed under the transaction."""


class CacheRetryExhausted(CacheError, ProviderRetryExhausted):
    """Retries exhausted for a transient error."""


class CacheRateLimitError(CacheRetryExhausted):
    """The server refused more work, after all retries exhausted."""


class CacheServerError(CacheRetryExhausted):
    """The service failed, after all retries exhausted."""


class CacheTimeoutError(CacheRetryExhausted):
    """Timed out, after all retries exhausted."""


class CacheConnectionError(CacheRetryExhausted):
    """Could not reach the service, after all retries exhausted."""


class CacheUnexpectedError(CacheError):
    """Anything we didn't anticipate. Always check `provider_error`."""
