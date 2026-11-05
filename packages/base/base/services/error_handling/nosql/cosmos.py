"""Azure SDK errors, as the Cosmos DB family."""

from base.models.definitions.error_handling import FailureKind
from base.models.view_models.exceptions.nosql import (
    NoSqlAuthError,
    NoSqlBadRequestError,
    NoSqlConflictError,
    NoSqlConnectionError,
    NoSqlNotFoundError,
    NoSqlRateLimitError,
    NoSqlServerError,
    NoSqlTimeoutError,
    NoSqlUnexpectedError,
)
from base.models.view_models.exceptions.provider import ProviderError
from base.services.error_handling.azure import AzureSDKErrorHandler

_ERROR_MAP: dict[FailureKind, type[ProviderError]] = {
    FailureKind.RATE_LIMIT: NoSqlRateLimitError,
    FailureKind.SERVER: NoSqlServerError,
    FailureKind.TIMEOUT: NoSqlTimeoutError,
    FailureKind.CONNECTION: NoSqlConnectionError,
    FailureKind.BAD_REQUEST: NoSqlBadRequestError,
    FailureKind.AUTH: NoSqlAuthError,
    FailureKind.NOT_FOUND: NoSqlNotFoundError,
    FailureKind.CONFLICT: NoSqlConflictError,
    FailureKind.UNKNOWN: NoSqlUnexpectedError,
}


class CosmosErrorHandler(AzureSDKErrorHandler):
    """Cosmos DB calls, raising the Cosmos DB family.

    Cosmos reports failures as `CosmosHttpResponseError`, which derives from
    the same `HttpResponseError` every other Azure client raises, so the status
    code reading is shared rather than repeated."""

    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        return _ERROR_MAP[kind]
