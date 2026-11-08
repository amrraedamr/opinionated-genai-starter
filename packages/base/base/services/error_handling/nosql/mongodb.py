"""PyMongo errors, as the document store family."""

from base.tools.errors import MissingExtraError

try:
    from pymongo.errors import (
        AutoReconnect,
        BulkWriteError,
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

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
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
from base.services.error_handling import AbstractErrorHandler

_UNAUTHORIZED_CODES: frozenset[int] = frozenset({13, 18, 8000})
"""Mongo reports a rejected credential as an `OperationFailure` carrying a
code: Unauthorized, AuthenticationFailed, and Atlas's own variant."""

_DUPLICATE_KEY_CODES: frozenset[int] = frozenset({11000, 11001, 12582})
"""A unique index refusing a write, whichever operation carried it."""

_RETRYABLE_CODES: frozenset[int] = frozenset({6, 7, 89, 134, 262, 9001})
"""Codes the driver retries that reach us as a bare `OperationFailure`.

HostUnreachable, HostNotFound, NetworkTimeout, ExceededTimeLimit,
SocketException, and a read concern whose majority has not caught up yet. The
driver raises a `ConnectionFailure` subclass for the rest of its retryable
set, so only these need the code to be read; without it they land among the
commands the server refused and are answered once instead of again.

They read as `SERVER` because the server or the path to it is what failed,
not the command.
"""

_FAILURE_MAP: dict[type[PyMongoError], FailureKind] = {
    # Order matters only for readability; the MRO walk resolves the subclass.
    NetworkTimeout: FailureKind.TIMEOUT,
    ServerSelectionTimeoutError: FailureKind.TIMEOUT,
    WaitQueueTimeoutError: FailureKind.TIMEOUT,
    # Alone among the timeouts, this one is the caller's own `maxTimeMS` spent
    # rather than the transport giving up, so it is the request that failed.
    ExecutionTimeout: FailureKind.BAD_REQUEST,
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
        """Resolves on the exception type first, then on the server's own code.

        The type carries the verdict for anything the driver raised itself,
        which is why it is read first. What the server refused arrives as one
        `OperationFailure` whatever the reason, so the code is the only thing
        separating a credential it rejected, or a host it could not reach,
        from a command it genuinely would not run.
        """
        for cls in type(error).__mro__:
            if cls in _FAILURE_MAP:
                return _FAILURE_MAP[cls]
        if isinstance(error, BulkWriteError):
            # A batch reports its own code 65 and carries the real reason per
            # write, so a duplicate in a batch reads like a duplicate alone.
            if any(
                write.get("code") in _DUPLICATE_KEY_CODES
                for write in error.details.get("writeErrors", [])
            ):
                return FailureKind.CONFLICT
        if isinstance(error, OperationFailure):
            if error.code in _UNAUTHORIZED_CODES:
                return FailureKind.AUTH
            if error.code in _RETRYABLE_CODES:
                return FailureKind.SERVER
            # Everything else the server refused is the caller's fault.
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
