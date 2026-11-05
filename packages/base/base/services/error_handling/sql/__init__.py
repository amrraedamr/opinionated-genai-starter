"""SQLAlchemy errors, as the relational database family."""

from base.models.definitions.error_handling import FailureKind
from base.models.view_models.exceptions.provider import ProviderError
from base.models.view_models.exceptions.sql import (
    SQLAuthError,
    SQLBadRequestError,
    SQLConflictError,
    SQLConnectionError,
    SQLNotFoundError,
    SQLRateLimitError,
    SQLServerError,
    SQLTimeoutError,
    SQLUnexpectedError,
)
from base.services.error_handling.sqlalchemy import SQLAlchemyErrorHandler

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: SQLRateLimitError,
    FailureKind.SERVER: SQLServerError,
    FailureKind.TIMEOUT: SQLTimeoutError,
    FailureKind.CONNECTION: SQLConnectionError,
    FailureKind.BAD_REQUEST: SQLBadRequestError,
    FailureKind.AUTH: SQLAuthError,
    FailureKind.NOT_FOUND: SQLNotFoundError,
    FailureKind.CONFLICT: SQLConflictError,
    FailureKind.UNKNOWN: SQLUnexpectedError,
}


class SQLErrorHandler(SQLAlchemyErrorHandler):
    """Relational calls, raising the SQL family.

    Safe to retry because every method it covers opens its own session, so a
    retry enters a new transaction rather than resuming a failed one, which
    SQLAlchemy does not allow. Methods taking a caller's statement are not
    covered by default: this layer cannot tell whether repeating one is safe.
    """

    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        return _ERROR_MAP[kind]
