"""How PyMongo's exceptions are read, as the document store family.

Two decisions carry the weight here. `ExecutionTimeout` is the caller's own
`maxTimeMS` running out, so repeating it would not help and might apply the
work twice — it must not retry, unlike every other timeout. And a handful of
transport failures arrive as a bare `OperationFailure` carrying only a code;
without reading it they land among the commands the server refused and are
answered once instead of again.
"""

from __future__ import annotations

import pymongo.errors
import pytest
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
    WriteError,
)

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.nosql import (
    NoSqlAuthError,
    NoSqlBadRequestError,
    NoSqlConflictError,
    NoSqlConnectionError,
    NoSqlError,
    NoSqlServerError,
    NoSqlTimeoutError,
    NoSqlUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderRetryExhausted
from base.services.error_handling.nosql.mongodb import MongoErrorHandler
from tests.fakes import subclasses_of
from tests.unit.error_handling.support import build_sdk_error, domain_class_for

RETRYABLE_CODES: tuple[int, ...] = (6, 7, 89, 134, 262, 9001)
AUTH_CODES: tuple[int, ...] = (13, 18, 8000)


@pytest.fixture
def handler() -> MongoErrorHandler:
    return MongoErrorHandler(retries=1, base_wait=0.0)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (NetworkTimeout("slow"), FailureKind.TIMEOUT),
        (ServerSelectionTimeoutError("no primary"), FailureKind.TIMEOUT),
        (WaitQueueTimeoutError("pool full"), FailureKind.TIMEOUT),
        (NotPrimaryError("stepped down"), FailureKind.CONNECTION),
        (AutoReconnect("socket closed"), FailureKind.CONNECTION),
        (ConnectionFailure("refused"), FailureKind.CONNECTION),
        (ConfigurationError("bad uri"), FailureKind.BAD_REQUEST),
    ],
)
def test_a_driver_raised_error_resolves_on_its_type(
    handler: MongoErrorHandler, error: PyMongoError, kind: FailureKind
) -> None:
    assert handler.failure_kind(error) is kind


def test_a_duplicate_key_is_a_conflict(handler: MongoErrorHandler) -> None:
    error = DuplicateKeyError("E11000 duplicate key", code=11000)
    assert handler.classify(error) == (NoSqlConflictError, False)


def test_a_stepping_down_primary_is_retried(handler: MongoErrorHandler) -> None:
    domain_cls, retryable = handler.classify(NotPrimaryError("not master"))
    assert (domain_cls, retryable) == (NoSqlConnectionError, True)


def test_a_cancelled_statement_is_not_retried(handler: MongoErrorHandler) -> None:
    """`maxTimeMS` ran out; the server may already have done the work."""
    error = ExecutionTimeout("operation exceeded time limit", code=50)
    assert handler.failure_kind(error) is FailureKind.BAD_REQUEST
    assert handler.classify(error) == (NoSqlBadRequestError, False)


def test_a_cancelled_statement_is_the_one_timeout_that_is_not_a_timeout(
    handler: MongoErrorHandler,
) -> None:
    timeouts = (NetworkTimeout("a"), ServerSelectionTimeoutError("b"))
    assert all(handler.classify(error)[1] for error in timeouts)
    assert handler.classify(ExecutionTimeout("c", code=50))[1] is False


@pytest.mark.parametrize("code", RETRYABLE_CODES)
def test_a_transport_code_the_server_reported_is_retried(
    handler: MongoErrorHandler, code: int
) -> None:
    error = OperationFailure("the path to the server failed", code=code)
    assert handler.failure_kind(error) is FailureKind.SERVER
    assert handler.classify(error) == (NoSqlServerError, True)


@pytest.mark.parametrize("code", AUTH_CODES)
def test_a_rejected_credential_is_an_auth_failure(
    handler: MongoErrorHandler, code: int
) -> None:
    error = OperationFailure("not authorized", code=code)
    assert handler.failure_kind(error) is FailureKind.AUTH
    assert handler.classify(error) == (NoSqlAuthError, False)


def test_a_command_the_server_refused_is_the_callers_fault(
    handler: MongoErrorHandler,
) -> None:
    error = OperationFailure("unknown operator $nope", code=2)
    assert handler.classify(error) == (NoSqlBadRequestError, False)


def test_an_operation_failure_with_no_code_is_the_callers_fault(
    handler: MongoErrorHandler,
) -> None:
    error = OperationFailure("refused")
    assert error.code is None
    assert handler.classify(error) == (NoSqlBadRequestError, False)


def test_the_exception_type_is_read_before_the_code(
    handler: MongoErrorHandler,
) -> None:
    """`ExecutionTimeout` is an `OperationFailure`; its type must win."""
    error = ExecutionTimeout("exceeded time limit", code=262)
    assert error.code in RETRYABLE_CODES
    assert handler.failure_kind(error) is FailureKind.BAD_REQUEST


def test_a_write_error_is_the_callers_fault(handler: MongoErrorHandler) -> None:
    assert handler.classify(WriteError("bad document", code=121)) == (
        NoSqlBadRequestError,
        False,
    )


def test_an_sdk_error_outside_the_map_is_unknown(
    handler: MongoErrorHandler,
) -> None:
    error = pymongo.errors.InvalidOperation("cursor already closed")
    assert handler.failure_kind(error) is FailureKind.UNKNOWN
    assert handler.classify(error) == (NoSqlUnexpectedError, False)


def test_a_wrapped_error_carries_the_servers_own_code(
    handler: MongoErrorHandler,
) -> None:
    error = DuplicateKeyError("E11000", code=11000)
    wrapped = handler.wrap(error, NoSqlConflictError)
    assert wrapped.status_code == 11000
    assert wrapped.provider_error is error


def test_a_wrapped_error_without_a_code_reports_none(
    handler: MongoErrorHandler,
) -> None:
    wrapped = handler.wrap(ConfigurationError("bad uri"), NoSqlBadRequestError)
    assert wrapped.status_code is None


def test_an_exhausted_error_carries_the_attempts_and_the_code(
    handler: MongoErrorHandler,
) -> None:
    error = OperationFailure("host unreachable", code=6)
    wrapped = handler.wrap_exhausted(error, 3)
    assert isinstance(wrapped, NoSqlServerError)
    assert (wrapped.attempts, wrapped.status_code) == (3, 6)
    assert "after 3 attempts" in str(wrapped)


def test_something_the_sdk_never_raised_becomes_the_unexpected_error(
    handler: MongoErrorHandler,
) -> None:
    error = ValueError("bson would not decode")
    wrapped = handler.wrap_unexpected(error)
    assert isinstance(wrapped, NoSqlUnexpectedError)
    assert wrapped.provider_error is error


def test_a_retried_timeout_exhausts_into_the_timeout_class(
    handler: MongoErrorHandler,
) -> None:
    wrapped = handler.wrap_exhausted(NetworkTimeout("slow"), 2)
    assert isinstance(wrapped, NoSqlTimeoutError)


def test_every_sdk_exception_maps_to_a_sane_kind() -> None:
    """Walks the SDK's whole tree so a class it adds later is covered too."""
    handler = MongoErrorHandler()
    covered = 0
    for cls in subclasses_of(PyMongoError):
        error = build_sdk_error(cls)
        if error is None:
            continue
        covered += 1
        kind = handler.failure_kind(error)
        assert isinstance(kind, FailureKind)
        domain_cls, retryable = handler.classify(error)
        assert retryable is (kind in RETRYABLE_FAILURES)
        assert domain_cls is domain_class_for(handler, kind)
        assert issubclass(domain_cls, NoSqlError)
        if retryable:
            assert issubclass(domain_cls, ProviderRetryExhausted)
            assert handler.wrap_exhausted(error, 2).attempts == 2
        else:
            assert isinstance(handler.wrap(error, domain_cls), domain_cls)
    assert covered >= 20


def test_the_sdk_base_class_is_what_the_handler_catches() -> None:
    assert MongoErrorHandler().provider_error_type is PyMongoError
