"""Azure AI Search errors, as the vector search family."""

from base.models.definitions.error_handling import FailureKind
from base.models.view_models.exceptions.provider import ProviderError
from base.models.view_models.exceptions.vector_search import (
    VectorSearchAuthError,
    VectorSearchBadRequestError,
    VectorSearchConflictError,
    VectorSearchConnectionError,
    VectorSearchNotFoundError,
    VectorSearchRateLimitError,
    VectorSearchServerError,
    VectorSearchTimeoutError,
    VectorSearchUnexpectedError,
)
from base.services.error_handling.azure import AzureSDKErrorHandler

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: VectorSearchRateLimitError,
    FailureKind.SERVER: VectorSearchServerError,
    FailureKind.TIMEOUT: VectorSearchTimeoutError,
    FailureKind.CONNECTION: VectorSearchConnectionError,
    FailureKind.BAD_REQUEST: VectorSearchBadRequestError,
    FailureKind.AUTH: VectorSearchAuthError,
    FailureKind.NOT_FOUND: VectorSearchNotFoundError,
    FailureKind.CONFLICT: VectorSearchConflictError,
    FailureKind.UNKNOWN: VectorSearchUnexpectedError,
}


class AzureSearchErrorHandler(AzureSDKErrorHandler):
    """Azure AI Search calls, raising the vector search family."""

    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        return _ERROR_MAP[kind]
