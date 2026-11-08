from base.tools.errors import MissingExtraError

try:
    from fastapi import Depends
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "server", "fastapi", feature="Bearer authentication dependencies"
    ) from exc

from uuid import UUID

from base.models.definitions.errors import AuthErrors
from base.services.auth.models import Principal
from base.services.auth.tokens import TokenVerifier
from base.tools.errors import raise_error

http_bearer = HTTPBearer()
http_bearer_dependency = Depends(http_bearer)


async def current_principal(
    credentials: HTTPAuthorizationCredentials = http_bearer_dependency,
) -> Principal:
    """Resolve the caller from token claims."""
    return TokenVerifier().verify(credentials.credentials)


principal_dependency = Depends(current_principal)


async def current_user_id(principal: Principal = principal_dependency) -> UUID:
    """The caller's own id, which a verified token need not carry.

    Layered rather than folded into `current_principal`: an identity
    provider's subject is rarely a UUID, and a route that only authenticates,
    or that resolves the caller by issuer and subject instead, is right to
    accept one. A route scoping rows by user id refuses here rather than
    querying for `user_id IS NULL`.
    """
    if principal.user_id is None:
        raise_error(AuthErrors.USER_TOKEN_INCOMPLETE)
    return principal.user_id


user_id_dependency = Depends(current_user_id)
