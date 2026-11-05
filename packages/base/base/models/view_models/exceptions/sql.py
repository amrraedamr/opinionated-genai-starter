"""Domain errors every relational backend raises. No provider SDK is named here."""

from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class SQLError(ProviderError):
    """Base class for all relational database errors."""


class SQLConfigError(SQLError, ProviderConfigError):
    """Invalid configuration. Trying again will not help."""


class SQLBadRequestError(SQLError):
    """Malformed statement, bad value, or an unsupported operation."""


class SQLAuthError(SQLError):
    """Bad credentials or insufficient permissions."""


class SQLNotFoundError(SQLError):
    """The table or relation does not exist."""


class SQLConflictError(SQLError):
    """A constraint rejected the write — a duplicate key, or a foreign key.

    Stores catch this to tell a collision from a genuine failure, so it stays
    distinct from a transaction the server rolled back for contention, which
    is retried and reported as a server failure.
    """


class SQLRetryExhausted(SQLError, ProviderRetryExhausted):
    """Retries exhausted for a transient error."""


class SQLRateLimitError(SQLRetryExhausted):
    """Throttled by the database, after all retries exhausted."""


class SQLServerError(SQLRetryExhausted):
    """The database failed or rolled the transaction back, after all retries."""


class SQLTimeoutError(SQLRetryExhausted):
    """Timed out, after all retries exhausted."""


class SQLConnectionError(SQLRetryExhausted):
    """Could not reach the database, after all retries exhausted."""


class SQLUnexpectedError(SQLError):
    """Anything we didn't anticipate. Always check `provider_error`."""
