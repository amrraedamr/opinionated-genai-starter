"""Azure SDK errors, as the blob family."""

from base.models.definitions.error_handling import FailureKind
from base.models.view_models.exceptions.blob import (
    BlobAuthError,
    BlobBadRequestError,
    BlobConflictError,
    BlobConnectionError,
    BlobNotFoundError,
    BlobRateLimitError,
    BlobServerError,
    BlobTimeoutError,
    BlobUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderError
from base.services.error_handling.azure import AzureSDKErrorHandler

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: BlobRateLimitError,
    FailureKind.SERVER: BlobServerError,
    FailureKind.TIMEOUT: BlobTimeoutError,
    FailureKind.CONNECTION: BlobConnectionError,
    FailureKind.BAD_REQUEST: BlobBadRequestError,
    FailureKind.AUTH: BlobAuthError,
    FailureKind.NOT_FOUND: BlobNotFoundError,
    FailureKind.CONFLICT: BlobConflictError,
    FailureKind.UNKNOWN: BlobUnexpectedError,
}


class AzureBlobErrorHandler(AzureSDKErrorHandler):
    """Azure blob calls, raising the blob family."""

    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        return _ERROR_MAP[kind]
