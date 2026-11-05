"""SQLAlchemy errors, as the vector search family."""

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
from base.services.error_handling.sqlalchemy import SQLAlchemyErrorHandler

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


class PgVectorErrorHandler(SQLAlchemyErrorHandler):
    """pgvector calls, raising the vector search family.

    Safe to retry because every statement here is idempotent — DDL guarded by
    `IF NOT EXISTS`, writes upserted with `ON CONFLICT DO UPDATE`, deletes by
    id — and each method opens its own transaction, so a retry starts a fresh
    one rather than resuming a failed session.
    """

    def error_for(self, kind: FailureKind) -> type[ProviderError]:
        return _ERROR_MAP[kind]
