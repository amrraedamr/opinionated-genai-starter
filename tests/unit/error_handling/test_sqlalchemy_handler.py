"""How SQLAlchemy's exceptions are read, where the SQLSTATE decides first.

A transaction the server rolled back for contention arrives as an
`OperationalError`, exactly like a dropped connection does, and only the
server's own five-character code separates them. So `40001`, `40P01` and
`55P03` are read before the exception type and retried, while `57014` — the
statement a `statement_timeout` cancelled — deliberately is not: a timeout
does not say whether the server finished the work.

asyncpg carries its SQLSTATE as a class attribute, which is the only way a
unit test can build one; psycopg2's `pgcode` is read-only and set by the
server, so a hand-built psycopg2 error reports nothing.
"""

from __future__ import annotations

import asyncpg
import psycopg2
import pytest
from sqlalchemy.exc import (
    DatabaseError,
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

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.provider import ProviderRetryExhausted
from base.models.view_models.exceptions.sql import (
    SQLBadRequestError,
    SQLConflictError,
    SQLConnectionError,
    SQLError,
    SQLNotFoundError,
    SQLServerError,
    SQLTimeoutError,
    SQLUnexpectedError,
)
from base.models.view_models.exceptions.vector_search import (
    VectorSearchConflictError,
    VectorSearchError,
    VectorSearchServerError,
)
from base.services.error_handling.sql import SQLErrorHandler
from base.services.error_handling.vector_search.pgvector import PgVectorErrorHandler
from tests.fakes import subclasses_of
from tests.unit.error_handling.support import build_sdk_error, domain_class_for

CONTENTION_SQLSTATES: tuple[tuple[str, type[Exception]], ...] = (
    ("40001", asyncpg.exceptions.SerializationError),
    ("40P01", asyncpg.exceptions.DeadlockDetectedError),
    ("55P03", asyncpg.exceptions.LockNotAvailableError),
)


def dbapi_error(
    cls: type[DBAPIError],
    orig: Exception | None = None,
    *,
    connection_invalidated: bool = False,
) -> DBAPIError:
    """A SQLAlchemy wrapper around a driver error, as the engine builds one."""
    return cls(
        "SELECT 1",
        {},
        orig or Exception("the driver said no"),
        connection_invalidated=connection_invalidated,
    )


@pytest.fixture
def handler() -> SQLErrorHandler:
    return SQLErrorHandler(retries=1, base_wait=0.0)


@pytest.mark.parametrize(("sqlstate", "driver_error"), CONTENTION_SQLSTATES)
def test_a_transaction_the_server_rolled_back_is_retried(
    handler: SQLErrorHandler, sqlstate: str, driver_error: type[Exception]
) -> None:
    error = dbapi_error(OperationalError, driver_error("the server gave up"))
    assert error.orig.sqlstate == sqlstate
    assert handler.failure_kind(error) is FailureKind.SERVER
    assert handler.classify(error) == (SQLServerError, True)


def test_a_cancelled_statement_is_not_retried(handler: SQLErrorHandler) -> None:
    """`57014` is `statement_timeout`; the work may already have landed."""
    error = dbapi_error(
        OperationalError, asyncpg.exceptions.QueryCanceledError("canceling statement")
    )
    assert error.orig.sqlstate == "57014"
    assert handler.failure_kind(error) is FailureKind.BAD_REQUEST
    assert handler.classify(error) == (SQLBadRequestError, False)


def test_a_unique_violation_is_a_conflict(handler: SQLErrorHandler) -> None:
    error = dbapi_error(
        IntegrityError, asyncpg.exceptions.UniqueViolationError("duplicate key")
    )
    assert error.orig.sqlstate == "23505"
    assert handler.classify(error) == (SQLConflictError, False)


def test_a_foreign_key_violation_is_a_conflict(handler: SQLErrorHandler) -> None:
    error = dbapi_error(
        IntegrityError,
        asyncpg.exceptions.ForeignKeyViolationError("no such parent"),
    )
    assert handler.failure_kind(error) is FailureKind.CONFLICT


def test_an_integrity_error_is_never_retried(handler: SQLErrorHandler) -> None:
    _, retryable = handler.classify(dbapi_error(IntegrityError))
    assert retryable is False


def test_a_contention_code_outranks_the_exception_type(
    handler: SQLErrorHandler,
) -> None:
    """An `IntegrityError` carrying `40001` is still the server rolling back."""
    error = dbapi_error(
        IntegrityError, asyncpg.exceptions.SerializationError("could not serialize")
    )
    assert handler.failure_kind(error) is FailureKind.SERVER


@pytest.mark.parametrize(
    ("cls", "kind"),
    [
        (SQLAlchemyTimeoutError, FailureKind.TIMEOUT),
        (DisconnectionError, FailureKind.CONNECTION),
        (NoSuchTableError, FailureKind.NOT_FOUND),
    ],
)
def test_an_error_the_engine_raised_itself_resolves_on_its_type(
    handler: SQLErrorHandler, cls: type[SQLAlchemyError], kind: FailureKind
) -> None:
    assert handler.failure_kind(cls("no")) is kind


@pytest.mark.parametrize(
    ("cls", "kind"),
    [
        (InterfaceError, FailureKind.CONNECTION),
        (ProgrammingError, FailureKind.BAD_REQUEST),
        (DataError, FailureKind.BAD_REQUEST),
        (NotSupportedError, FailureKind.BAD_REQUEST),
        (InternalError, FailureKind.SERVER),
    ],
)
def test_a_driver_error_resolves_on_its_wrapper_type(
    handler: SQLErrorHandler, cls: type[DBAPIError], kind: FailureKind
) -> None:
    assert handler.failure_kind(dbapi_error(cls)) is kind


def test_a_pool_that_invalidated_the_connection_is_a_connection_failure(
    handler: SQLErrorHandler,
) -> None:
    error = dbapi_error(OperationalError, connection_invalidated=True)
    assert handler.failure_kind(error) is FailureKind.CONNECTION
    assert handler.classify(error) == (SQLConnectionError, True)


def test_an_operational_error_the_pool_kept_is_the_callers_fault(
    handler: SQLErrorHandler,
) -> None:
    """Only the driver knows which; without its verdict the statement is blamed."""
    error = dbapi_error(OperationalError)
    assert error.connection_invalidated is False
    assert handler.classify(error) == (SQLBadRequestError, False)


def test_a_cancelled_statement_on_an_invalidated_connection_is_retried(
    handler: SQLErrorHandler,
) -> None:
    """The pool's verdict is read after the type walk, so it wins over `57014`."""
    error = dbapi_error(
        OperationalError,
        asyncpg.exceptions.QueryCanceledError("canceling"),
        connection_invalidated=True,
    )
    assert handler.failure_kind(error) is FailureKind.CONNECTION
    assert handler.classify(error) == (SQLConnectionError, True)


def test_a_hand_built_psycopg2_error_reports_no_sqlstate(
    handler: SQLErrorHandler,
) -> None:
    """`pgcode` is read-only and server-set, so a test cannot fake one."""
    orig = psycopg2.errors.SerializationFailure("could not serialize")
    assert getattr(orig, "pgcode", None) is None
    error = dbapi_error(OperationalError, orig)
    assert handler.failure_kind(error) is FailureKind.BAD_REQUEST


def test_a_sqlstate_is_read_from_whichever_name_the_driver_uses(
    handler: SQLErrorHandler,
) -> None:
    class Psycopg2Style(Exception):
        pgcode = "40P01"

    error = dbapi_error(OperationalError, Psycopg2Style("deadlock detected"))
    assert handler.failure_kind(error) is FailureKind.SERVER


def test_a_driver_with_no_sqlstate_at_all_falls_back_to_the_type(
    handler: SQLErrorHandler,
) -> None:
    error = dbapi_error(DataError, Exception("sqlite has no code"))
    assert handler.failure_kind(error) is FailureKind.BAD_REQUEST


def test_an_error_with_no_original_driver_error_falls_back_to_the_type(
    handler: SQLErrorHandler,
) -> None:
    error = DatabaseError("SELECT 1", {}, None)
    assert error.orig is None
    assert handler.failure_kind(error) is FailureKind.UNKNOWN


def test_an_error_outside_the_map_is_unknown(handler: SQLErrorHandler) -> None:
    from sqlalchemy.exc import ArgumentError

    error = ArgumentError("bad mapping")
    assert handler.failure_kind(error) is FailureKind.UNKNOWN
    assert handler.classify(error) == (SQLUnexpectedError, False)


def test_a_missing_table_is_never_retried(handler: SQLErrorHandler) -> None:
    assert handler.classify(NoSuchTableError("widgets")) == (SQLNotFoundError, False)


def test_a_pool_timeout_is_retried(handler: SQLErrorHandler) -> None:
    assert handler.classify(SQLAlchemyTimeoutError("pool exhausted")) == (
        SQLTimeoutError,
        True,
    )


def test_a_wrapped_error_drops_the_statement_that_may_carry_values(
    handler: SQLErrorHandler,
) -> None:
    error = IntegrityError(
        "INSERT INTO users (email) VALUES ('secret@example.com')",
        {"email": "secret@example.com"},
        Exception("duplicate key value violates unique constraint"),
    )
    wrapped = handler.wrap(error, SQLConflictError)
    assert "duplicate key value" in str(wrapped)
    assert "secret@example.com" not in str(wrapped)


def test_an_exhausted_error_carries_the_attempts(
    handler: SQLErrorHandler,
) -> None:
    error = dbapi_error(
        OperationalError, asyncpg.exceptions.DeadlockDetectedError("deadlock")
    )
    wrapped = handler.wrap_exhausted(error, 3)
    assert isinstance(wrapped, SQLServerError)
    assert wrapped.attempts == 3
    assert "deadlock" in str(wrapped)


def test_something_the_engine_never_raised_becomes_the_unexpected_error(
    handler: SQLErrorHandler,
) -> None:
    error = ValueError("the row did not convert")
    wrapped = handler.wrap_unexpected(error)
    assert isinstance(wrapped, SQLUnexpectedError)
    assert wrapped.provider_error is error


def test_pgvector_reads_the_same_codes_but_raises_the_vector_family() -> None:
    sql = SQLErrorHandler()
    pgvector = PgVectorErrorHandler()
    deadlock = dbapi_error(
        OperationalError, asyncpg.exceptions.DeadlockDetectedError("deadlock")
    )
    duplicate = dbapi_error(
        IntegrityError, asyncpg.exceptions.UniqueViolationError("duplicate")
    )
    assert sql.failure_kind(deadlock) is pgvector.failure_kind(deadlock)
    assert pgvector.classify(deadlock) == (VectorSearchServerError, True)
    assert pgvector.classify(duplicate) == (VectorSearchConflictError, False)


@pytest.mark.parametrize(
    ("handler_cls", "family"),
    [(SQLErrorHandler, SQLError), (PgVectorErrorHandler, VectorSearchError)],
)
def test_every_engine_exception_maps_to_a_sane_kind(
    handler_cls: type, family: type[Exception]
) -> None:
    """Walks SQLAlchemy's whole tree so a class it adds later is covered too."""
    handler = handler_cls()
    covered = 0
    for cls in subclasses_of(SQLAlchemyError):
        error = build_sdk_error(cls)
        if error is None:
            continue
        covered += 1
        kind = handler.failure_kind(error)
        assert isinstance(kind, FailureKind)
        domain_cls, retryable = handler.classify(error)
        assert retryable is (kind in RETRYABLE_FAILURES)
        assert domain_cls is domain_class_for(handler, kind)
        assert issubclass(domain_cls, family)
        if retryable:
            assert issubclass(domain_cls, ProviderRetryExhausted)
            assert handler.wrap_exhausted(error, 2).attempts == 2
        else:
            assert isinstance(handler.wrap(error, domain_cls), domain_cls)
    assert covered >= 40


def test_the_engine_base_class_is_what_the_handler_catches() -> None:
    assert SQLErrorHandler().provider_error_type is SQLAlchemyError
    assert PgVectorErrorHandler().provider_error_type is SQLAlchemyError
