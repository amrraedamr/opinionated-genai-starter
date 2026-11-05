"""OpenAI SDK errors, as the embedding family."""

from base.models.view_models.exceptions.embedding import (
    EmbeddingAuthError,
    EmbeddingBadRequestError,
    EmbeddingConflictError,
    EmbeddingConnectionError,
    EmbeddingNotFoundError,
    EmbeddingRateLimitError,
    EmbeddingServerError,
    EmbeddingTimeoutError,
    EmbeddingUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderError
from base.services.error_handling import FailureKind
from base.services.error_handling.openai import OpenAISDKErrorHandler

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: EmbeddingRateLimitError,
    FailureKind.SERVER: EmbeddingServerError,
    FailureKind.TIMEOUT: EmbeddingTimeoutError,
    FailureKind.CONNECTION: EmbeddingConnectionError,
    FailureKind.BAD_REQUEST: EmbeddingBadRequestError,
    FailureKind.AUTH: EmbeddingAuthError,
    FailureKind.NOT_FOUND: EmbeddingNotFoundError,
    FailureKind.CONFLICT: EmbeddingConflictError,
    FailureKind.UNKNOWN: EmbeddingUnexpectedError,
}


class OpenAIEmbeddingErrorHandler(OpenAISDKErrorHandler):
    """OpenAI embedding calls, raising the embedding family.

    No content-filter arm: an embedding request is not generated text, so the
    filter never rejects one.
    """

    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        return _ERROR_MAP[kind]
