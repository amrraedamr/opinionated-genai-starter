from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials

from base.models.orm.user import UserDTO
from base.services.auth import AuthService
from base.services.auth.auths import http_bearer

http_bearer_dependency = Depends(http_bearer)


async def avalidate_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = http_bearer_dependency,
) -> UserDTO:
    return await AuthService().avalidate_user(credentials)


validate_user_dependency = Depends(avalidate_user)
