"""Retry policy and domain mapping, shared by every provider.

The loop, the backoff and the logging are the same whoever is behind the call.
What differs is which exception type the SDK raises, how it classifies, and
where it hides the status code and the retry hint — so those are the only
things an implementation supplies.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import final

from base.models.view_models.llm_exception import (
    LLMError,
    LLMRetryExhausted,
    LLMUnexpectedError,
)
from base.services.logging.console import ConsoleLoggerService


class AbstractErrorHandler[E: Exception](ABC):
    """Runs a provider call, retrying what is worth retrying.

    Callers get a result or an `LLMError`; a raw SDK exception never escapes.
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
        """The SDK's base exception. Anything else is unexpected."""

    @abstractmethod
    def classify(self, error: E) -> tuple[type[LLMError], bool]:
        """Resolve an SDK error to `(domain class, is retryable)`."""

    @abstractmethod
    def identify(self, error: E) -> tuple[int | None, str | None]:
        """Pull `(status code, request id)` out of an SDK error."""

    def retry_after(self, error: E) -> float | None:
        """Seconds the provider asked us to wait, when it said so."""
        del error
        return None

    def wrap(self, error: E, domain_cls: type[LLMError]) -> LLMError:
        """Build the domain error for a non-retryable failure."""
        status_code, request_id = self.identify(error)
        return domain_cls.from_provider(
            error, status_code=status_code, request_id=request_id
        )

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
                # Intentional boundary: map unknown failures to our domain error.
                raise LLMUnexpectedError.from_provider(exc) from exc

        assert last_error is not None
        raise self._wrap_exhausted(last_error, attempts=max_attempts) from last_error

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
                # Intentional boundary: map unknown failures to our domain error.
                raise LLMUnexpectedError.from_provider(exc) from exc

        assert last_error is not None
        raise self._wrap_exhausted(last_error, attempts=max_attempts) from last_error

    def _wrap_exhausted(self, error: E, attempts: int) -> LLMRetryExhausted:
        domain_cls, _ = self.classify(error)
        assert issubclass(domain_cls, LLMRetryExhausted), (
            f"Retryable error {type(error).__name__} mapped to non-retry-exhausted "
            f"domain class {domain_cls.__name__}"
        )
        status_code, request_id = self.identify(error)
        return domain_cls(
            f"{domain_cls.__name__} after {attempts} attempts: {error}",
            attempts=attempts,
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    def _compute_wait(self, error: E, attempt: int) -> float:
        """Respect the provider's own hint, else jittered exponential backoff."""
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
