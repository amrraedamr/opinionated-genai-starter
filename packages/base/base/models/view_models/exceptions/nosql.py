"""Domain errors every document store adapter raises. No provider SDK is named here."""

from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class NoSqlError(ProviderError):
    """Base class for all document store errors."""


class NoSqlConfigError(NoSqlError, ProviderConfigError):
    """Invalid configuration. Trying again will not help."""


class NoSqlBadRequestError(NoSqlError):
    """Malformed query, bad document, or an unsupported operation."""


class NoSqlAuthError(NoSqlError):
    """Bad credentials or insufficient permissions."""


class NoSqlNotFoundError(NoSqlError):
    """The database, container or document does not exist."""


class NoSqlConflictError(NoSqlError):
    """A document with that key already exists, or an etag no longer matches."""


class NoSqlRetryExhausted(NoSqlError, ProviderRetryExhausted):
    """Retries exhausted for a transient error."""


class NoSqlRateLimitError(NoSqlRetryExhausted):
    """Throttled by the store, after all retries exhausted."""


class NoSqlServerError(NoSqlRetryExhausted):
    """The service failed, after all retries exhausted."""


class NoSqlTimeoutError(NoSqlRetryExhausted):
    """Timed out, after all retries exhausted."""


class NoSqlConnectionError(NoSqlRetryExhausted):
    """Could not reach the service, after all retries exhausted."""


class NoSqlUnexpectedError(NoSqlError):
    """Anything we didn't anticipate. Always check `provider_error`."""
