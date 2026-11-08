"""Redis errors, as the cache family."""

from base.tools.errors import MissingExtraError

try:
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
        TryAgainError,
        WatchError,
    )
    from redis.exceptions import (
        ConnectionError as RedisConnectionError,
    )
    from redis.exceptions import (
        TimeoutError as RedisTimeoutError,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("redis", "redis", feature="RedisErrorHandler") from exc

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.cache import (
    CacheAuthError,
    CacheBadRequestError,
    CacheConflictError,
    CacheConnectionError,
    CacheNotFoundError,
    CacheRateLimitError,
    CacheServerError,
    CacheTimeoutError,
    CacheUnexpectedError,
)
from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.services.error_handling import AbstractErrorHandler

_FAILURE_MAP: dict[type[RedisError], FailureKind] = {
    # Authentication derives from ConnectionError, so it must resolve first.
    AuthenticationError: FailureKind.AUTH,
    AuthorizationError: FailureKind.AUTH,
    NoPermissionError: FailureKind.AUTH,
    BusyLoadingError: FailureKind.SERVER,
    ClusterDownError: FailureKind.SERVER,
    ReadOnlyError: FailureKind.SERVER,
    OutOfMemoryError: FailureKind.RATE_LIMIT,
    MaxConnectionsError: FailureKind.RATE_LIMIT,
    RedisTimeoutError: FailureKind.TIMEOUT,
    RedisConnectionError: FailureKind.CONNECTION,
    WatchError: FailureKind.CONFLICT,
    NoScriptError: FailureKind.NOT_FOUND,
    DataError: FailureKind.BAD_REQUEST,
    # A cluster moving a slot answers TRYAGAIN, and retrying is the
    # documented remedy. It subclasses ResponseError, so it has to be
    # named ahead of it.
    TryAgainError: FailureKind.SERVER,
    ResponseError: FailureKind.BAD_REQUEST,
}

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: CacheRateLimitError,
    FailureKind.SERVER: CacheServerError,
    FailureKind.TIMEOUT: CacheTimeoutError,
    FailureKind.CONNECTION: CacheConnectionError,
    FailureKind.BAD_REQUEST: CacheBadRequestError,
    FailureKind.AUTH: CacheAuthError,
    FailureKind.NOT_FOUND: CacheNotFoundError,
    FailureKind.CONFLICT: CacheConflictError,
    FailureKind.UNKNOWN: CacheUnexpectedError,
}


class RedisErrorHandler(AbstractErrorHandler[RedisError]):
    """Redis calls, raising the cache family.

    A loading replica and a cluster mid-failover both answer again shortly, so
    they count as server failures rather than permanent ones. Running out of
    memory or connections is the server refusing more work, which is closer to
    being throttled than to being broken.
    """

    @property
    def provider_error_type(self) -> type[RedisError]:
        return RedisError

    def failure_kind(self, error: RedisError) -> FailureKind:
        """Walks the MRO so a subclass resolves like its parent."""
        for cls in type(error).__mro__:
            if cls in _FAILURE_MAP:
                return _FAILURE_MAP[cls]
        return FailureKind.UNKNOWN

    def classify(self, error: RedisError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return _ERROR_MAP[kind], kind in RETRYABLE_FAILURES

    def wrap(self, error: RedisError, domain_cls: type[Exception]) -> Exception:
        return domain_cls(str(error), provider_error=error)

    def wrap_exhausted(self, error: RedisError, attempts: int) -> Exception:
        domain_cls, _ = self.classify(error)
        assert issubclass(domain_cls, ProviderRetryExhausted), (
            f"Retryable error {type(error).__name__} mapped to non-retry-exhausted "
            f"domain class {domain_cls.__name__}"
        )
        return domain_cls(
            f"{domain_cls.__name__} after {attempts} attempts: {error}",
            attempts=attempts,
            provider_error=error,
        )

    def wrap_unexpected(self, error: Exception) -> Exception:
        return CacheUnexpectedError.from_provider(error)
