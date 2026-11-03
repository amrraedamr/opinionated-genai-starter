import uuid

from pydantic import field_validator, model_validator

from base.models.view_models import BaseModel
from base.tools.validators import validate_name, validate_password


class CreateUserResponse(BaseModel):
    user_id: uuid.UUID


class GetUserResponse(BaseModel):
    user_id: uuid.UUID
    email: str
    first_name: str | None = None
    last_name: str | None = None


class EditUserRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    current_password: str | None = None
    new_password: str | None = None

    @field_validator("first_name", "last_name")
    @classmethod
    def _validate_names(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_name(value)

    @field_validator("new_password")
    @classmethod
    def _validate_new_password(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_password(value)

    @model_validator(mode="after")
    def _password_fields_together(self) -> "EditUserRequest":
        if (self.current_password is None) != (self.new_password is None):
            raise ValueError(
                "current_password and new_password must be provided together"
            )
        return self


class EditUserResponse(BaseModel):
    pass


class DeleteUserResponse(BaseModel):
    pass
