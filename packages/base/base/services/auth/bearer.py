from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from base.services.auth.models import Principal
from base.services.auth.tokens import TokenVerifier

http_bearer = HTTPBearer()
http_bearer_dependency = Depends(http_bearer)


async def current_principal(
    credentials: HTTPAuthorizationCredentials = http_bearer_dependency,
) -> Principal:
    """Resolve the caller from token claims."""
    return TokenVerifier().verify(credentials.credentials)


principal_dependency = Depends(current_principal)
