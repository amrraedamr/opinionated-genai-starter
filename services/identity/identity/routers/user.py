from fastapi import APIRouter

from base.models.definitions.errors import AuthErrors
from base.models.view_models import BaseResponse
from base.tools.errors import raise_error
from base.tools.security import PasswordHashingService
from identity.models.orm.user import UserDTO
from identity.models.view_models.user import (
    DeleteUserResponse,
    EditUserRequest,
    EditUserResponse,
    GetUserResponse,
)
from identity.services.auth.dependencies import validate_user_dependency
from identity.stores.user import UserStore

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=BaseResponse[GetUserResponse])
async def get_user(
    current_user: UserDTO = validate_user_dependency,
) -> BaseResponse[GetUserResponse]:
    return BaseResponse(
        data=GetUserResponse(
            user_id=current_user.id,
            email=current_user.email,
            first_name=current_user.first_name,
            last_name=current_user.last_name,
        )
    )


@router.patch("/me", response_model=BaseResponse[EditUserResponse])
async def edit_user(
    request: EditUserRequest,
    current_user: UserDTO = validate_user_dependency,
) -> BaseResponse[EditUserResponse]:
    updates = request.serialize(
        exclude_unset=True,
        exclude_none=True,
        by_alias=False,
        exclude={"current_password", "new_password"},
    )
    current_user.update_from_dict(updates)

    if request.new_password is not None:
        assert request.current_password is not None
        if not current_user.password_hash:
            raise_error(AuthErrors.PASSWORD_NOT_SET)
        if not PasswordHashingService.verify(
            request.current_password, current_user.password_hash
        ):
            raise_error(AuthErrors.USER_UNAUTHORIZED)
        current_user.password_hash = PasswordHashingService.hash(request.new_password)

    await UserStore().aedit(current_user.id, current_user)
    return BaseResponse(data=EditUserResponse())


@router.delete("/me", response_model=BaseResponse[DeleteUserResponse])
async def delete_user(
    current_user: UserDTO = validate_user_dependency,
) -> BaseResponse[DeleteUserResponse]:
    await UserStore().adelete(current_user.id)
    return BaseResponse(data=DeleteUserResponse())
