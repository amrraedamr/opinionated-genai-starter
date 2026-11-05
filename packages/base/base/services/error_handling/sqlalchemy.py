"""Reading SQLAlchemy's exceptions, without deciding what callers see.

Shared by anything that reaches a database through SQLAlchemy. Retrying one of
these is only safe when the statement is idempotent and the method opens its
own transaction, because a failed session cannot be reused — a caller that
retries must re-enter the transaction, not resume it.
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

from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.services.error_handling import (
    RETRYABLE_FAILURES,
    AbstractErrorHandler,
    FailureKind,
)

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


class SQLAlchemyErrorHandler(AbstractErrorHandler[SQLAlchemyError], ABC):
    """The SQLAlchemy half of error handling. Subclasses choose the domain."""

    @abstractmethod
    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        """The domain error class this kind of failure becomes."""

    @property
    def provider_error_type(self) -> type[SQLAlchemyError]:
        return SQLAlchemyError

    def failure_kind(self, error: SQLAlchemyError) -> FailureKind:
        """Walks the MRO so a subclass resolves like its parent.

        `OperationalError` is deliberately not in the map. It covers both a
        dropped connection and a statement the server refused, and only the
        driver knows which — so the pool's own verdict is used instead, and
        anything it did not invalidate is treated as permanent rather than
        retried into the same wall.
        """
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
    def _message(error: SQLAlchemyError) -> str:
        """The driver's message without the statement, which may carry values."""
        return str(getattr(error, "orig", None) or error)
