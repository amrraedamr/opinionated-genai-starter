import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from base.tools.errors import MissingExtraError

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        ConflictError,
        InternalServerError,
        NotFoundError,
        OpenAIError,
        PermissionDeniedError,
        RateLimitError,
        UnprocessableEntityError,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="OpenAIErrorHandler") from exc

from base.models.view_models.llm_exception import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConflictError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMRetryExhausted,
    LLMServerError,
    LLMTimeoutError,
    LLMUnexpectedError,
)
from base.services.logging.console import ConsoleLoggerService

T = TypeVar("T")


_ERROR_MAP: dict[type[OpenAIError], tuple[type[LLMError], bool]] = {
    # Retryable
    RateLimitError: (LLMRateLimitError, True),
    InternalServerError: (LLMServerError, True),
    APITimeoutError: (LLMTimeoutError, True),
    APIConnectionError: (LLMConnectionError, True),
    # Non-retryable
    BadRequestError: (LLMBadRequestError, False),
    UnprocessableEntityError: (LLMBadRequestError, False),
    AuthenticationError: (LLMAuthError, False),
    PermissionDeniedError: (LLMAuthError, False),
    NotFoundError: (LLMNotFoundError, False),
    ConflictError: (LLMConflictError, False),
}


class OpenAIErrorHandler:
    """
    Centralizes OpenAI SDK error classification, retry policy, and domain
    exception mapping.

    Services call .run() / .arun() with a callable and receive either the
    result or a raised domain LLMError. Callers never see raw OpenAI SDK
    exceptions.
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
        self._logger = ConsoleLoggerService()

    # Public runners

    def run(
        self,
        func: Callable[..., T],
        *args,
        retry: bool = True,
        **kwargs,
    ) -> T:
        max_attempts: int = (self._retries + 1) if retry else 1
        last_error: OpenAIError | None = None

        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)

            except OpenAIError as e:
                domain_cls, is_retryable = self._classify(e)

                if is_retryable and attempt + 1 < max_attempts:
                    last_error = e
                    wait = self._compute_wait(e, attempt)
                    self._log_retry(e, attempt, max_attempts, wait)
                    time.sleep(wait)
                    continue

                if is_retryable:
                    last_error = e
                    self._log_exhausted(e, attempt, max_attempts)
                    break

                raise self._wrap(e, domain_cls) from e

            except Exception as e:
                # Intentional boundary: map unknown failures to our domain error.
                raise LLMUnexpectedError.from_openai(e) from e

        assert last_error is not None
        raise self._wrap_exhausted(last_error, attempts=max_attempts) from last_error

    async def arun(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        retry: bool = True,
        **kwargs,
    ) -> T:
        max_attempts: int = (self._retries + 1) if retry else 1
        last_error: OpenAIError | None = None

        for attempt in range(max_attempts):
            try:
                return await func(*args, **kwargs)

            except OpenAIError as e:
                domain_cls, is_retryable = self._classify(e)

                if is_retryable and attempt + 1 < max_attempts:
                    last_error = e
                    wait = self._compute_wait(e, attempt)
                    self._log_retry(e, attempt, max_attempts, wait)
                    await asyncio.sleep(wait)
                    continue

                if is_retryable:
                    last_error = e
                    self._log_exhausted(e, attempt, max_attempts)
                    break

                raise self._wrap(e, domain_cls) from e

            except Exception as e:
                # Intentional boundary: map unknown failures to our domain error.
                raise LLMUnexpectedError.from_openai(e) from e

        assert last_error is not None
        raise self._wrap_exhausted(last_error, attempts=max_attempts) from last_error

    # Classification

    def _classify(self, error: OpenAIError) -> tuple[type[LLMError], bool]:
        """
        Resolve an OpenAI SDK error to (domain_class, is_retryable).

        Walks the MRO so subclasses of mapped exceptions resolve to the same
        domain type as their parent. Unknown OpenAI errors fall through to
        LLMUnexpectedError (non-retryable).

        Special case: a ``BadRequestError`` whose ``code`` (or response
        body) marks the request as a content-filter rejection is routed
        to :class:`LLMContentFilterError` instead of the generic bad
        request mapping. Azure OpenAI surfaces both pre-flight prompt
        rejections and post-generation response filters as HTTP 400
        with ``code == "content_filter"``.
        """
        if isinstance(error, BadRequestError) and self._is_content_filter(error):
            return LLMContentFilterError, False

        for cls in type(error).__mro__:
            if cls in _ERROR_MAP:
                return _ERROR_MAP[cls]
        return LLMUnexpectedError, False

    @staticmethod
    def _is_content_filter(error: BadRequestError) -> bool:
        code = getattr(error, "code", None)
        if code == "content_filter":
            return True
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict) and inner.get("code") == "content_filter":
                return True
        message = getattr(error, "message", None) or str(error)
        return (
            "content management policy" in message.lower()
            or "content_filter" in message.lower()
        )

    def _wrap(self, error: OpenAIError, domain_cls: type[LLMError]) -> LLMError:
        """Wrap a non-retryable error in its domain class."""
        if domain_cls is LLMContentFilterError:
            return self._wrap_content_filter(error)
        return domain_cls.from_openai(error)

    @staticmethod
    def _wrap_content_filter(error: OpenAIError) -> LLMContentFilterError:
        status_code: int | None = getattr(error, "status_code", None)
        request_id: str | None = None
        if isinstance(error, APIStatusError):
            request_id = error.response.headers.get("x-request-id")
        message = getattr(error, "message", None) or str(error)
        return LLMContentFilterError(
            message,
            incomplete_details=None,
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    def _wrap_exhausted(
        self,
        error: OpenAIError,
        attempts: int,
    ) -> LLMRetryExhausted:
        """Wrap a retryable error that exhausted all attempts."""
        domain_cls, _ = self._classify(error)

        # Retryable errors all map to LLMRetryExhausted subclasses.
        assert issubclass(domain_cls, LLMRetryExhausted), (
            f"Retryable error {type(error).__name__} mapped to non-retry-exhausted "
            f"domain class {domain_cls.__name__}"
        )

        status_code: int | None = None
        request_id: str | None = None
        if isinstance(error, APIStatusError):
            status_code = error.status_code
            request_id = error.response.headers.get("x-request-id")

        return domain_cls(
            f"{domain_cls.__name__} after {attempts} attempts: {error}",
            attempts=attempts,
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    # Wait / backoff

    def _compute_wait(self, error: OpenAIError, attempt: int) -> float:
        """Respect Retry-After on 429, else jittered exponential backoff."""
        if isinstance(error, RateLimitError):
            retry_after = self._parse_retry_after(error)
            if retry_after is not None:
                return retry_after
        return self._backoff(attempt)

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter. attempt is 0-indexed."""
        raw = min(self._max_wait, self._base_wait * (2**attempt))
        return random.uniform(0, raw)

    def _parse_retry_after(self, error: OpenAIError) -> float | None:
        """Extract Retry-After header from a RateLimitError, if present."""
        response = getattr(error, "response", None)
        if response is None:
            return None
        header = response.headers.get("retry-after")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None

    # Logging

    def _log_retry(
        self,
        error: OpenAIError,
        attempt: int,
        max_attempts: int,
        wait: float,
    ) -> None:
        self._logger.warning(
            f"Attempt {attempt + 1}/{max_attempts} failed: "
            f"{type(error).__name__} — retrying in {wait:.2f}s"
        )

    def _log_exhausted(
        self,
        error: OpenAIError,
        attempt: int,
        max_attempts: int,
    ) -> None:
        self._logger.warning(
            f"Attempt {attempt + 1}/{max_attempts} failed: "
            f"{type(error).__name__} — retries exhausted"
        )
