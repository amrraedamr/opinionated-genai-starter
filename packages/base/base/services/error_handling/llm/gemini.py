"""Google GenAI errors, as the LLM family.

The SDK reports every failure as an `APIError` carrying the HTTP status, so
the status decides rather than the exception type — the same arrangement the
Azure reader uses, and for the same reason.
"""

from http import HTTPStatus

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind
from base.models.view_models.exceptions.llm import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConflictError,
    LLMConnectionError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnexpectedError,
)
from base.models.view_models.exceptions.provider import (
    ProviderError,
    ProviderRetryExhausted,
)
from base.services.error_handling import AbstractErrorHandler
from base.tools.errors import MissingExtraError

try:
    from google.genai import errors as genai_errors
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "gemini", "google-genai", feature="Gemini error handling"
    ) from exc

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: LLMRateLimitError,
    FailureKind.SERVER: LLMServerError,
    FailureKind.TIMEOUT: LLMTimeoutError,
    FailureKind.CONNECTION: LLMConnectionError,
    FailureKind.BAD_REQUEST: LLMBadRequestError,
    FailureKind.AUTH: LLMAuthError,
    FailureKind.NOT_FOUND: LLMNotFoundError,
    FailureKind.CONFLICT: LLMConflictError,
    FailureKind.UNKNOWN: LLMUnexpectedError,
}

_STATUS_KINDS: tuple[tuple[int, FailureKind], ...] = (
    (HTTPStatus.TOO_MANY_REQUESTS, FailureKind.RATE_LIMIT),
    (HTTPStatus.REQUEST_TIMEOUT, FailureKind.TIMEOUT),
    (HTTPStatus.GATEWAY_TIMEOUT, FailureKind.TIMEOUT),
    (HTTPStatus.UNAUTHORIZED, FailureKind.AUTH),
    (HTTPStatus.FORBIDDEN, FailureKind.AUTH),
    (HTTPStatus.NOT_FOUND, FailureKind.NOT_FOUND),
    (HTTPStatus.CONFLICT, FailureKind.CONFLICT),
)


class GeminiErrorHandler(AbstractErrorHandler[genai_errors.APIError]):
    """Gemini calls, raising the LLM family."""

    @property
    def provider_error_type(self) -> type[genai_errors.APIError]:
        return genai_errors.APIError

    def failure_kind(self, error: genai_errors.APIError) -> FailureKind:
        """Reads the HTTP status, which is the only thing that separates these.

        `RESOURCE_EXHAUSTED` and `UNAVAILABLE` arrive as a `ClientError` and a
        `ServerError`, but so does every other refusal at that status class.
        """
        code = getattr(error, "code", None)
        if not isinstance(code, int):
            return FailureKind.UNKNOWN
        for status, kind in _STATUS_KINDS:
            if code == status:
                return kind
        if code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            return FailureKind.SERVER
        if code >= HTTPStatus.BAD_REQUEST:
            return FailureKind.BAD_REQUEST
        return FailureKind.UNKNOWN

    def classify(self, error: genai_errors.APIError) -> tuple[type[Exception], bool]:
        kind = self.failure_kind(error)
        return _ERROR_MAP[kind], kind in RETRYABLE_FAILURES

    def wrap(
        self, error: genai_errors.APIError, domain_cls: type[Exception]
    ) -> Exception:
        return domain_cls(
            self._message(error),
            status_code=getattr(error, "code", None),
            provider_error=error,
        )

    def wrap_exhausted(self, error: genai_errors.APIError, attempts: int) -> Exception:
        domain_cls, _ = self.classify(error)
        assert issubclass(domain_cls, ProviderRetryExhausted), (
            f"Retryable error {type(error).__name__} mapped to non-retry-exhausted "
            f"domain class {domain_cls.__name__}"
        )
        return domain_cls(
            f"{domain_cls.__name__} after {attempts} attempts: {self._message(error)}",
            attempts=attempts,
            status_code=getattr(error, "code", None),
            provider_error=error,
        )

    def wrap_unexpected(self, error: Exception) -> Exception:
        return LLMUnexpectedError.from_provider(error)

    @staticmethod
    def _message(error: genai_errors.APIError) -> str:
        """The provider's message plus its own status name, where it sent one."""
        message = getattr(error, "message", None) or str(error)
        status = getattr(error, "status", None)
        return f"{status}: {message}" if status else str(message)
