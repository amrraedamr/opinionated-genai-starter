"""Azure SDK errors, as the Key Vault family."""

from base.models.definitions.error_handling import FailureKind
from base.models.view_models.exceptions.keyvault import (
    KeyVaultAuthError,
    KeyVaultBadRequestError,
    KeyVaultConflictError,
    KeyVaultConnectionError,
    KeyVaultNotFoundError,
    KeyVaultRateLimitError,
    KeyVaultServerError,
    KeyVaultTimeoutError,
    KeyVaultUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderError
from base.services.error_handling.azure import AzureSDKErrorHandler

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: KeyVaultRateLimitError,
    FailureKind.SERVER: KeyVaultServerError,
    FailureKind.TIMEOUT: KeyVaultTimeoutError,
    FailureKind.CONNECTION: KeyVaultConnectionError,
    FailureKind.BAD_REQUEST: KeyVaultBadRequestError,
    FailureKind.AUTH: KeyVaultAuthError,
    FailureKind.NOT_FOUND: KeyVaultNotFoundError,
    FailureKind.CONFLICT: KeyVaultConflictError,
    FailureKind.UNKNOWN: KeyVaultUnexpectedError,
}


class AzureKeyVaultErrorHandler(AzureSDKErrorHandler):
    """Key Vault calls, raising the Key Vault family."""

    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        return _ERROR_MAP[kind]
