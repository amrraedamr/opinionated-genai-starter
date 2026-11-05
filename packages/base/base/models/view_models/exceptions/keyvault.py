"""Domain errors every secret store adapter raises. No provider SDK is named here."""

from base.models.view_models.exceptions.provider import (
    ProviderConfigError,
    ProviderError,
    ProviderRetryExhausted,
)


class KeyVaultError(ProviderError):
    """Base class for all secret store errors."""


class KeyVaultConfigError(KeyVaultError, ProviderConfigError):
    """Invalid configuration. Trying again will not help."""


class KeyVaultBadRequestError(KeyVaultError):
    """Malformed secret name or value."""


class KeyVaultAuthError(KeyVaultError):
    """Bad credentials or insufficient permissions."""


class KeyVaultNotFoundError(KeyVaultError):
    """The secret or vault does not exist."""


class KeyVaultConflictError(KeyVaultError):
    """The secret is in a conflicting state, such as soft-deleted."""


class KeyVaultRetryExhausted(KeyVaultError, ProviderRetryExhausted):
    """Retries exhausted for a transient error."""


class KeyVaultRateLimitError(KeyVaultRetryExhausted):
    """Throttled by the vault, after all retries exhausted."""


class KeyVaultServerError(KeyVaultRetryExhausted):
    """The service failed, after all retries exhausted."""


class KeyVaultTimeoutError(KeyVaultRetryExhausted):
    """Timed out, after all retries exhausted."""


class KeyVaultConnectionError(KeyVaultRetryExhausted):
    """Could not reach the service, after all retries exhausted."""


class KeyVaultUnexpectedError(KeyVaultError):
    """Anything we didn't anticipate. Always check `provider_error`."""
