"""How Redis's exceptions are read, as the cache family.

Two orderings matter. `AuthenticationError` derives from `ConnectionError`, so
a rejected password must resolve as auth rather than being retried forever as
a dropped socket. And running out of memory or connections is the server
refusing more work, which is read as throttling rather than a broken server.
"""

from __future__ import annotations

import pytest
from redis.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BusyLoadingError,
    ClusterDownError,
    DataError,
    MaxConnectionsError,
    NoPermissionError,
    NoScriptError,
    OutOfMemoryError,
    ReadOnlyError,
    RedisError,
    ResponseError,
    WatchError,
)
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
)
from redis.exceptions import (
    TimeoutError as RedisTimeoutError,
)

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.cache import (
    CacheAuthError,
    CacheBadRequestError,
    CacheConflictError,
    CacheConnectionError,
    CacheError,
    CacheNotFoundError,
    CacheRateLimitError,
    CacheServerError,
    CacheTimeoutError,
    CacheUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderRetryExhausted
from base.services.error_handling.cache.redis import RedisErrorHandler
from tests.fakes import subclasses_of
from tests.unit.error_handling.support import build_sdk_error, domain_class_for

MAPPED: tuple[tuple[RedisError, FailureKind, type[Exception]], ...] = (
    (AuthenticationError("wrong password"), FailureKind.AUTH, CacheAuthError),
    (AuthorizationError("no acl"), FailureKind.AUTH, CacheAuthError),
    (NoPermissionError("denied"), FailureKind.AUTH, CacheAuthError),
    (BusyLoadingError("loading dataset"), FailureKind.SERVER, CacheServerError),
    (ClusterDownError("failing over"), FailureKind.SERVER, CacheServerError),
    (ReadOnlyError("replica"), FailureKind.SERVER, CacheServerError),
    (OutOfMemoryError("OOM"), FailureKind.RATE_LIMIT, CacheRateLimitError),
    (MaxConnectionsError("pool full"), FailureKind.RATE_LIMIT, CacheRateLimitError),
    (RedisTimeoutError("timed out"), FailureKind.TIMEOUT, CacheTimeoutError),
    (RedisConnectionError("refused"), FailureKind.CONNECTION, CacheConnectionError),
    (WatchError("key changed"), FailureKind.CONFLICT, CacheConflictError),
    (NoScriptError("NOSCRIPT"), FailureKind.NOT_FOUND, CacheNotFoundError),
    (DataError("cannot encode"), FailureKind.BAD_REQUEST, CacheBadRequestError),
    (ResponseError("WRONGTYPE"), FailureKind.BAD_REQUEST, CacheBadRequestError),
)


@pytest.fixture
def handler() -> RedisErrorHandler:
    return RedisErrorHandler(retries=1, base_wait=0.0)


@pytest.mark.parametrize(("error", "kind", "domain_cls"), MAPPED)
def test_an_error_resolves_to_its_kind_and_class(
    handler: RedisErrorHandler,
    error: RedisError,
    kind: FailureKind,
    domain_cls: type[Exception],
) -> None:
    assert handler.failure_kind(error) is kind
    assert handler.classify(error) == (domain_cls, kind in RETRYABLE_FAILURES)


def test_a_rejected_password_is_auth_rather_than_a_dropped_socket(
    handler: RedisErrorHandler,
) -> None:
    """`AuthenticationError` derives from `ConnectionError`; auth must win."""
    error = AuthenticationError("invalid password")
    assert isinstance(error, RedisConnectionError)
    assert handler.classify(error) == (CacheAuthError, False)


def test_a_loading_replica_is_retried(handler: RedisErrorHandler) -> None:
    assert handler.classify(BusyLoadingError("loading")) == (CacheServerError, True)


def test_a_cluster_mid_failover_is_retried(handler: RedisErrorHandler) -> None:
    assert handler.classify(ClusterDownError("CLUSTERDOWN")) == (
        CacheServerError,
        True,
    )


def test_running_out_of_memory_is_treated_as_throttling(
    handler: RedisErrorHandler,
) -> None:
    assert handler.classify(OutOfMemoryError("OOM")) == (CacheRateLimitError, True)


def test_a_lost_optimistic_lock_is_never_retried(
    handler: RedisErrorHandler,
) -> None:
    assert handler.classify(WatchError("changed")) == (CacheConflictError, False)


def test_a_wrong_type_command_is_never_retried(handler: RedisErrorHandler) -> None:
    assert handler.classify(ResponseError("WRONGTYPE")) == (
        CacheBadRequestError,
        False,
    )


def test_an_error_outside_the_map_is_unknown(handler: RedisErrorHandler) -> None:
    from redis.exceptions import LockError

    error = LockError("not acquired")
    assert handler.failure_kind(error) is FailureKind.UNKNOWN
    assert handler.classify(error) == (CacheUnexpectedError, False)


def test_a_cluster_slot_migration_is_not_retried(
    handler: RedisErrorHandler,
) -> None:
    """`TRYAGAIN` is a `ResponseError`, so it lands among refused commands."""
    from redis.exceptions import TryAgainError

    error = TryAgainError("TRYAGAIN")
    assert handler.failure_kind(error) is FailureKind.BAD_REQUEST
    assert handler.classify(error) == (CacheBadRequestError, False)


def test_a_master_that_went_down_is_retried(handler: RedisErrorHandler) -> None:
    from redis.exceptions import MasterDownError

    assert handler.classify(MasterDownError("MASTERDOWN")) == (
        CacheServerError,
        True,
    )


def test_a_wrapped_error_keeps_the_drivers_message(
    handler: RedisErrorHandler,
) -> None:
    error = ResponseError("WRONGTYPE Operation against a key")
    wrapped = handler.wrap(error, CacheBadRequestError)
    assert "WRONGTYPE" in str(wrapped)
    assert wrapped.provider_error is error


def test_a_wrapped_error_reports_no_status_because_redis_sends_none(
    handler: RedisErrorHandler,
) -> None:
    wrapped = handler.wrap(DataError("bad value"), CacheBadRequestError)
    assert wrapped.status_code is None


def test_an_exhausted_error_carries_the_attempts(
    handler: RedisErrorHandler,
) -> None:
    error = BusyLoadingError("loading dataset in memory")
    wrapped = handler.wrap_exhausted(error, 3)
    assert isinstance(wrapped, CacheServerError)
    assert wrapped.attempts == 3
    assert "after 3 attempts" in str(wrapped)


def test_the_handler_offers_no_retry_hint(handler: RedisErrorHandler) -> None:
    assert handler.retry_after(BusyLoadingError("loading")) is None


def test_something_the_sdk_never_raised_becomes_the_unexpected_error(
    handler: RedisErrorHandler,
) -> None:
    error = ValueError("could not decode the reply")
    wrapped = handler.wrap_unexpected(error)
    assert isinstance(wrapped, CacheUnexpectedError)
    assert wrapped.provider_error is error


def test_every_sdk_exception_maps_to_a_sane_kind() -> None:
    """Walks the SDK's whole tree so a class it adds later is covered too."""
    handler = RedisErrorHandler()
    covered = 0
    for cls in subclasses_of(RedisError):
        error = build_sdk_error(cls)
        if error is None:
            continue
        covered += 1
        kind = handler.failure_kind(error)
        assert isinstance(kind, FailureKind)
        domain_cls, retryable = handler.classify(error)
        assert retryable is (kind in RETRYABLE_FAILURES)
        assert domain_cls is domain_class_for(handler, kind)
        assert issubclass(domain_cls, CacheError)
        if retryable:
            assert issubclass(domain_cls, ProviderRetryExhausted)
            assert handler.wrap_exhausted(error, 2).attempts == 2
        else:
            assert isinstance(handler.wrap(error, domain_cls), domain_cls)
    assert covered >= 25


def test_the_sdk_base_class_is_what_the_handler_catches() -> None:
    assert RedisErrorHandler().provider_error_type is RedisError
