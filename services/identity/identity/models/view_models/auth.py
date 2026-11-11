from pydantic import EmailStr, Field, field_validator

from base.models.definitions.tenancy import DEFAULT_TENANT
from base.models.view_models import BaseModel
from base.services.auth import AccessToken, RefreshToken, TokenPair
from base.tools.validators import validate_name, validate_password

__all__ = [
    "AccessToken",
    "LoginRequest",
    "LoginResponse",
    "OAuthExchangeRequest",
    "OAuthProvidersResponse",
    "RefreshRequest",
    "RefreshResponse",
    "RefreshToken",
    "RegisterRequest",
    "TokenPair",
]


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str | None = None
    last_name: str | None = None
    tenant_id: str = Field(default=DEFAULT_TENANT, min_length=1, max_length=255)

    @field_validator("first_name", "last_name")
    @classmethod
    def _validate_names(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_name(value)

    _validate_password = field_validator("password")(validate_password)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    access_token: AccessToken
    refresh_token: RefreshToken


class RefreshRequest(BaseModel):
    refresh_token: RefreshToken = Field(max_length=8192)


class RefreshResponse(BaseModel):
    access_token: AccessToken
    refresh_token: RefreshToken


class OAuthExchangeRequest(BaseModel):
    code: str = Field(min_length=1, max_length=2048)


class OAuthProvidersResponse(BaseModel):
    providers: list[str]
