"""PyMongo errors, as the document store family."""

from base.tools.errors import MissingExtraError

try:
    from pymongo.errors import (
        AutoReconnect,
        ConfigurationError,
        ConnectionFailure,
        DuplicateKeyError,
        ExecutionTimeout,
        NetworkTimeout,
        NotPrimaryError,
        OperationFailure,
        PyMongoError,
        ServerSelectionTimeoutError,
        WaitQueueTimeoutError,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("mongo", "pymongo", feature="MongoErrorHandler") from exc

from base.models.view_models.exceptions.nosql import (
    NoSqlAuthError,
    NoSqlBadRequestError,
    NoSqlConflictError,
    NoSqlConnectionError,
    NoSqlNotFoundError,
    NoSqlRateLimitError,
    NoSqlServerError,
    NoSqlTimeoutError,
    NoSqlUnexpectedError,
)
from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.services.error_handling import (
    RETRYABLE_FAILURES,
    AbstractErrorHandler,
    FailureKind,
)

_UNAUTHORIZED_CODES: frozenset[int] = frozenset({13, 18, 8000})
"""Mongo reports a rejected credential as an `OperationFailure` carrying a
code: Unauthorized, AuthenticationFailed, and Atlas's own variant."""

_FAILURE_MAP: dict[type[PyMongoError], FailureKind] = {
    # Order matters only for readability; the MRO walk resolves the subclass.
    NetworkTimeout: FailureKind.TIMEOUT,
    ServerSelectionTimeoutError: FailureKind.TIMEOUT,
    WaitQueueTimeoutError: FailureKind.TIMEOUT,
    ExecutionTimeout: FailureKind.TIMEOUT,
    DuplicateKeyError: FailureKind.CONFLICT,
    NotPrimaryError: FailureKind.CONNECTION,
    AutoReconnect: FailureKind.CONNECTION,
    ConnectionFailure: FailureKind.CONNECTION,
    ConfigurationError: FailureKind.BAD_REQUEST,
}

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: NoSqlRateLimitError,
    FailureKind.SERVER: NoSqlServerError,
    FailureKind.TIMEOUT: NoSqlTimeoutError,
    FailureKind.CONNECTION: NoSqlConnectionError,
    FailureKind.BAD_REQUEST: NoSqlBadRequestError,
    FailureKind.AUTH: NoSqlAuthError,
    FailureKind.NOT_FOUND: NoSqlNotFoundError,
    FailureKind.CONFLICT: NoSqlConflictError,
    FailureKind.UNKNOWN: NoSqlUnexpectedError,
}


class MongoErrorHandler(AbstractErrorHandler[PyMongoError]):
    """MongoDB calls, raising the document store family.

    A stepping-down primary raises `NotPrimaryError`, which is worth retrying:
    the driver finds the new primary and the next attempt lands on it.
    """

    @property
    def provider_error_type(self) -> type[PyMongoError]:
        return PyMongoError

    def failure_kind(self, error: PyMongoError) -> FailureKind:
        """Walks the MRO so a subclass resolves like its parent."""
        for cls in type(error).__mro__:
            if cls in _FAILURE_MAP:
                return _FAILURE_MAP[cls]
        if isinstance(error, OperationFailure):
            # Everything else the server refused is the caller's fault.
            if error.code in _UNAUTHORIZED_CODES:
                return FailureKind.AUTH
            return FailureKind.BAD_REQUEST
        return FailureKind.UNKNOWN

    def classify(self, error: PyMongoError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return _ERROR_MAP[kind], kind in RETRYABLE_FAILURES

    def wrap(self, error: PyMongoError, domain_cls: type[Exception]) -> Exception:
        return domain_cls(
            str(error), status_code=self._code(error), provider_error=error
        )

    def wrap_exhausted(self, error: PyMongoError, attempts: int) -> Exception:
        domain_cls, _ = self.classify(error)
        assert issubclass(domain_cls, ProviderRetryExhausted), (
            f"Retryable error {type(error).__name__} mapped to non-retry-exhausted "
            f"domain class {domain_cls.__name__}"
        )
        return domain_cls(
            f"{domain_cls.__name__} after {attempts} attempts: {error}",
            attempts=attempts,
            status_code=self._code(error),
            provider_error=error,
        )

    def wrap_unexpected(self, error: Exception) -> Exception:
        return NoSqlUnexpectedError.from_provider(error)

    @staticmethod
    def _code(error: PyMongoError) -> int | None:
        """Mongo's own error code, where it reports one."""
        code = getattr(error, "code", None)
        return int(code) if isinstance(code, int) else None
