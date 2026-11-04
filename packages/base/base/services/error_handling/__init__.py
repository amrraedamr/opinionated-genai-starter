"""Retrying a call that talks to something outside this process.

A model provider, an object store, a vector database, an identity provider —
anything whose failures split into "try again" and "this will never work".
The loop, the backoff and the logging are the same whatever is at the other
end. What differs is which exception the client raises, how it classifies, and
where it hides the status code and the retry hint.

Nothing here names a domain. An implementation decides which exceptions callers
see, so a store handler raises store errors and a model handler raises model
errors, both over one retry policy.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import final

from base.services.logging.console import ConsoleLoggerService


class FailureKind(StrEnum):
    """What went wrong, in terms no client or domain owns.

    An SDK handler decides which kind an exception is; a domain decides which
    error class a kind becomes. Stating it once is what lets one SDK serve
    several domains without repeating the classification.
    """

    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    BAD_REQUEST = "bad_request"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


RETRYABLE_FAILURES: frozenset[FailureKind] = frozenset(
    {
        FailureKind.RATE_LIMIT,
        FailureKind.SERVER,
        FailureKind.TIMEOUT,
        FailureKind.CONNECTION,
    }
)
"""Kinds worth trying again. A kind is retryable whoever raised it."""


class AbstractErrorHandler[E: Exception](ABC):
    """Runs a call, retrying what is worth retrying.

    Callers get a result or the domain error this handler wraps failures in; a
    raw client exception never escapes.
    """

    DEFAULT_RETRIES: int = 3
    DEFAULT_BASE_WAIT: float = 0.5
    DEFAULT_MAX_WAIT: float = 10.0

    def __init__(
        self,
        retries: int | None = None,
        base_wait: float | None = None,
        max_wait: float | None = None,
    ) -> None:
        self._retries: int = retries if retries is not None else self.DEFAULT_RETRIES
        self._base_wait: float = (
            base_wait if base_wait is not None else self.DEFAULT_BASE_WAIT
        )
        self._max_wait: float = (
            max_wait if max_wait is not None else self.DEFAULT_MAX_WAIT
        )
        self._logger: ConsoleLoggerService = ConsoleLoggerService()

    @property
    @abstractmethod
    def provider_error_type(self) -> type[E]:
        """The client's base exception. Anything else is unexpected."""

    @abstractmethod
    def classify(self, error: E) -> tuple[type[Exception], bool]:
        """Resolve a client error to `(domain class, is retryable)`."""

    @abstractmethod
    def wrap(self, error: E, domain_cls: type[Exception]) -> Exception:
        """Build the domain error for a failure that will not be retried."""

    @abstractmethod
    def wrap_exhausted(self, error: E, attempts: int) -> Exception:
        """Build the domain error for a retryable failure that ran out of tries."""

    @abstractmethod
    def wrap_unexpected(self, error: Exception) -> Exception:
        """Build the domain error for something the client never promised."""

    def retry_after(self, error: E) -> float | None:
        """Seconds the other end asked us to wait, when it said so.

        Most clients say nothing, so the default is nothing and the handler
        falls back to backoff.
        """
        del error
        return None

    @final
    def run[T](
        self,
        func: Callable[..., T],
        *args,
        retry: bool = True,
        **kwargs,
    ) -> T:
        """Run `func`, retrying transient failures."""
        max_attempts = (self._retries + 1) if retry else 1
        last_error: E | None = None

        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except self.provider_error_type as exc:
                domain_cls, retryable = self.classify(exc)
                if retryable and attempt + 1 < max_attempts:
                    last_error = exc
                    wait = self._compute_wait(exc, attempt)
                    self._log_retry(exc, attempt, max_attempts, wait)
                    time.sleep(wait)
                    continue
                if retryable:
                    last_error = exc
                    self._log_exhausted(exc, attempt, max_attempts)
                    break
                raise self.wrap(exc, domain_cls) from exc
            except Exception as exc:
                # Intentional boundary: nothing raw escapes this call.
                raise self.wrap_unexpected(exc) from exc

        assert last_error is not None
        raise self.wrap_exhausted(last_error, max_attempts) from last_error

    @final
    async def arun[T](
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        retry: bool = True,
        **kwargs,
    ) -> T:
        """Await `func`, retrying transient failures."""
        max_attempts = (self._retries + 1) if retry else 1
        last_error: E | None = None

        for attempt in range(max_attempts):
            try:
                return await func(*args, **kwargs)
            except self.provider_error_type as exc:
                domain_cls, retryable = self.classify(exc)
                if retryable and attempt + 1 < max_attempts:
                    last_error = exc
                    wait = self._compute_wait(exc, attempt)
                    self._log_retry(exc, attempt, max_attempts, wait)
                    await asyncio.sleep(wait)
                    continue
                if retryable:
                    last_error = exc
                    self._log_exhausted(exc, attempt, max_attempts)
                    break
                raise self.wrap(exc, domain_cls) from exc
            except Exception as exc:
                # Intentional boundary: nothing raw escapes this call.
                raise self.wrap_unexpected(exc) from exc

        assert last_error is not None
        raise self.wrap_exhausted(last_error, max_attempts) from last_error

    def _compute_wait(self, error: E, attempt: int) -> float:
        """Respect the other end's own hint, else jittered exponential backoff."""
        hinted = self.retry_after(error)
        return hinted if hinted is not None else self._backoff(attempt)

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter. `attempt` is 0-indexed."""
        raw = min(self._max_wait, self._base_wait * (2**attempt))
        return random.uniform(0, raw)

    def _log_retry(
        self, error: E, attempt: int, max_attempts: int, wait: float
    ) -> None:
        self._logger.warning(
            f"Attempt {attempt + 1}/{max_attempts} failed: "
            f"{type(error).__name__} — retrying in {wait:.2f}s"
        )

    def _log_exhausted(self, error: E, attempt: int, max_attempts: int) -> None:
        self._logger.warning(
            f"Attempt {attempt + 1}/{max_attempts} failed: "
            f"{type(error).__name__} — retries exhausted"
        )
