"""OpenAI SDK errors, as the LLM family."""

from base.tools.errors import MissingExtraError

try:
    from openai import BadRequestError, OpenAIError
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="OpenAIErrorHandler") from exc

from base.models.definitions.error_handling import FailureKind
from base.models.view_models.exceptions.llm import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConflictError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderError
from base.services.error_handling.openai import OpenAISDKErrorHandler

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


class OpenAIErrorHandler(OpenAISDKErrorHandler):
    """OpenAI generation calls, raising the LLM family."""

    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        return _ERROR_MAP[kind]

    def classify(self, error: OpenAIError) -> tuple[type[Exception], bool]:
        """A content-filter rejection arrives as a plain 400.

        Azure surfaces both prompt rejections and post-generation filters as
        HTTP 400 with `code == "content_filter"`, which is meaningful to a
        caller in a way an ordinary bad request is not.
        """
        if isinstance(error, BadRequestError) and self._is_content_filter(error):
            return LLMContentFilterError, False
        return super().classify(error)

    def wrap(self, error: OpenAIError, domain_cls: type[Exception]) -> Exception:
        if domain_cls is LLMContentFilterError:
            status_code, request_id = self._identify(error)
            return LLMContentFilterError(
                getattr(error, "message", None) or str(error),
                incomplete_details=None,
                status_code=status_code,
                request_id=request_id,
                provider_error=error,
            )
        return super().wrap(error, domain_cls)

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
