"""Reading SQLAlchemy's exceptions, without deciding what callers see.

Shared by anything that reaches a database through SQLAlchemy. Retrying one of
these is only safe when the statement is idempotent and the method opens its
own transaction, because a failed session cannot be reused — a caller that
retries must re-enter the transaction, not resume it.

Where the driver reports a SQLSTATE, that decides first. A transaction the
server rolled back for contention is the case the database itself tells you to
retry, and it is indistinguishable from a refused statement by exception type
alone.
"""

from abc import ABC, abstractmethod

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy.exc import (
        DataError,
        DBAPIError,
        DisconnectionError,
        IntegrityError,
        InterfaceError,
        InternalError,
        NoSuchTableError,
        NotSupportedError,
        OperationalError,
        ProgrammingError,
        SQLAlchemyError,
    )
    from sqlalchemy.exc import (
        TimeoutError as SQLAlchemyTimeoutError,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "sql", "sqlalchemy", feature="SQLAlchemy error handling"
    ) from exc

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.services.error_handling import AbstractErrorHandler

_FAILURE_MAP: dict[type[SQLAlchemyError], FailureKind] = {
    SQLAlchemyTimeoutError: FailureKind.TIMEOUT,
    DisconnectionError: FailureKind.CONNECTION,
    InterfaceError: FailureKind.CONNECTION,
    IntegrityError: FailureKind.CONFLICT,
    NoSuchTableError: FailureKind.NOT_FOUND,
    ProgrammingError: FailureKind.BAD_REQUEST,
    DataError: FailureKind.BAD_REQUEST,
    NotSupportedError: FailureKind.BAD_REQUEST,
    InternalError: FailureKind.SERVER,
}

_RETRYABLE_SQLSTATES: frozenset[str] = frozenset(
    {
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "55P03",  # lock_not_available
    }
)
"""Codes where the server aborted the transaction and says to start again.

Class 40 is PostgreSQL's transaction rollback class, and retrying is the
documented remedy rather than a workaround: a serialization failure resolves
once the competing transaction's result is visible, and a deadlock resolves
because the next attempt is unlikely to interleave the same way. MySQL reports
its own deadlock as `40001` too. `55P03` is the same situation reached through
a lock timeout.

They read as `SERVER` because the server is what aborted the work. The
alternative would be a failure kind of their own, which every domain would
then have to name.
"""


class SQLAlchemyErrorHandler(AbstractErrorHandler[SQLAlchemyError], ABC):
    """The SQLAlchemy half of error handling. Subclasses choose the domain."""

    @abstractmethod
    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        """The domain error class this kind of failure becomes."""

    @property
    def provider_error_type(self) -> type[SQLAlchemyError]:
        return SQLAlchemyError

    def failure_kind(self, error: SQLAlchemyError) -> FailureKind:
        """Resolves on the server's own code first, then on the exception type.

        A transaction the server rolled back for contention arrives as an
        `OperationalError` like a dropped connection does, and only the
        SQLSTATE separates them. Reading it first is what stops a deadlock —
        which the database tells us to retry — from being treated as a
        statement the server refused.

        `OperationalError` is otherwise deliberately not in the map. It covers
        both a dropped connection and a statement the server refused, and only
        the driver knows which, so the pool's own verdict is used instead. A
        cancelled statement (`57014`, which a `statement_timeout` raises) stays
        on that path on purpose: a timeout does not say whether the server
        finished the work, so repeating it could apply it twice.
        """
        if self._sqlstate(error) in _RETRYABLE_SQLSTATES:
            return FailureKind.SERVER
        for cls in type(error).__mro__:
            if cls in _FAILURE_MAP:
                return _FAILURE_MAP[cls]
        if isinstance(error, OperationalError):
            return (
                FailureKind.CONNECTION
                if self._connection_lost(error)
                else FailureKind.BAD_REQUEST
            )
        return FailureKind.UNKNOWN

    def classify(self, error: SQLAlchemyError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return self.error_for(kind), kind in RETRYABLE_FAILURES

    def wrap(self, error: SQLAlchemyError, domain_cls: type[Exception]) -> Exception:
        return domain_cls(self._message(error), provider_error=error)

    def wrap_exhausted(self, error: SQLAlchemyError, attempts: int) -> Exception:
        domain_cls, _ = self.classify(error)
        assert issubclass(domain_cls, ProviderRetryExhausted), (
            f"Retryable error {type(error).__name__} mapped to non-retry-exhausted "
            f"domain class {domain_cls.__name__}"
        )
        return domain_cls(
            f"{domain_cls.__name__} after {attempts} attempts: {self._message(error)}",
            attempts=attempts,
            provider_error=error,
        )

    def wrap_unexpected(self, error: Exception) -> Exception:
        return self.error_for(FailureKind.UNKNOWN).from_provider(error)

    @staticmethod
    def _connection_lost(error: SQLAlchemyError) -> bool:
        return bool(isinstance(error, DBAPIError) and error.connection_invalidated)

    @staticmethod
    def _sqlstate(error: SQLAlchemyError) -> str:
        """The server's five-character code, where the driver exposes one.

        psycopg2 calls it `pgcode`; psycopg3 and asyncpg call it `sqlstate`.
        SQLite has no equivalent, so this is empty there.
        """
        orig = getattr(error, "orig", None)
        if orig is None:
            return ""
        code = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
        return str(code) if code else ""

    @staticmethod
    def _message(error: SQLAlchemyError) -> str:
        """The driver's message without the statement, which may carry values."""
        return str(getattr(error, "orig", None) or error)
