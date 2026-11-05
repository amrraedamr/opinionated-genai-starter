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
import functools
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, final

from base.models.view_models.exceptions.provider import ProviderError
from base.services.logging.console import ConsoleLoggerService
from base.tools.attempt_trail import (
    mark_attempt,
    mark_attempt_failure,
    start_attempt_trail,
)


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

    @final
    def as_domain_error(self, error: Exception) -> Exception:
        """The domain error for a failure this handler did not run.

        For work a caller drove itself — a transaction it opened and statements
        it issued — where there is nothing to repeat but callers should still
        see one family rather than the client's own exception. A retryable
        failure is reported as having had the one attempt that was made.
        """
        if not isinstance(error, self.provider_error_type):
            return self.wrap_unexpected(error)
        domain_cls, retryable = self.classify(error)
        if retryable:
            return self.wrap_exhausted(error, 1)
        return self.wrap(error, domain_cls)

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
        start_attempt_trail()

        for attempt in range(max_attempts):
            mark_attempt()
            try:
                return func(*args, **kwargs)
            except self.provider_error_type as exc:
                mark_attempt_failure(exc)
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
            except ProviderError:
                # Already classified, by this handler or one further in. A
                # method that calls another through the same handler would
                # otherwise have its failure re-wrapped as unexpected, losing
                # what the inner call worked out and the attempts it spent.
                raise
            except Exception as exc:
                # Intentional boundary: nothing raw escapes this call.
                mark_attempt_failure(exc)
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
        start_attempt_trail()

        for attempt in range(max_attempts):
            mark_attempt()
            try:
                return await func(*args, **kwargs)
            except self.provider_error_type as exc:
                mark_attempt_failure(exc)
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
            except ProviderError:
                # Already classified, by this handler or one further in. A
                # method that calls another through the same handler would
                # otherwise have its failure re-wrapped as unexpected, losing
                # what the inner call worked out and the attempts it spent.
                raise
            except Exception as exc:
                # Intentional boundary: nothing raw escapes this call.
                mark_attempt_failure(exc)
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


def retrying[T](method: Callable[..., T]) -> Callable[..., T]:
    """Run a service method through the handler on its `_error_handler`.

    Fits a method that *is* the call: it builds a request, makes it, and shapes
    the reply. Retrying re-runs the whole body, so that body must be safe to
    repeat. It also means a malformed reply becomes a domain error rather than
    escaping as whatever the parsing raised.

    It does not fit a method that wraps a call in work that must happen once —
    recording usage, appending to a debug trace, emitting stream events. The
    LLM adapters are that shape, and call the handler around the request alone.

    Never use it on a generator either: calling one returns an iterator without
    doing any work, so the failure happens later, outside the retry loop.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        handler: AbstractErrorHandler[Any] = self._error_handler
        return handler.run(method, self, *args, **kwargs)

    return wrapper


def aretrying[T](method: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Await a service method through the handler on its `_error_handler`.

    The same rules as `retrying`: the body must be safe to repeat, and must not
    carry work that has to happen exactly once. Never wrap an async generator.
    """

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        handler: AbstractErrorHandler[Any] = self._error_handler
        return await handler.arun(method, self, *args, **kwargs)

    return wrapper


def classifying[T](method: Callable[..., T]) -> Callable[..., T]:
    """Run a method through the handler, but do not repeat it by default.

    For a call this layer cannot judge: one that carries a statement the caller
    wrote, where repeating it may apply it twice. Failures still become the
    domain's errors rather than the client's, so a caller catches one family
    whichever method it used, and decides for itself whether to try again by
    passing `retry=True`.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, retry: bool = False, **kwargs: Any) -> T:
        handler: AbstractErrorHandler[Any] = self._error_handler
        return handler.run(method, self, *args, retry=retry, **kwargs)

    return wrapper


def aclassifying[T](method: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Await a method through the handler without repeating it by default.

    The async counterpart of `classifying`, for a call carrying a statement
    this layer did not write.
    """

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, retry: bool = False, **kwargs: Any) -> T:
        handler: AbstractErrorHandler[Any] = self._error_handler
        return await handler.arun(method, self, *args, retry=retry, **kwargs)

    return wrapper
