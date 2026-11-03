"""Token and identity types."""

from typing import NewType
from uuid import UUID

from pydantic import BaseModel as PydanticBaseModel

from base.models.view_models import BaseModel

AccessToken = NewType("AccessToken", str)
RefreshToken = NewType("RefreshToken", str)


class TokenPair(BaseModel):
    access_token: AccessToken
    refresh_token: RefreshToken


class Principal(PydanticBaseModel):
    """The verified caller, reconstructed from token claims alone."""

    subject: str
    user_id: UUID | None = None
    email: str | None = None
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    issuer: str | None = None
    token_type: str | None = None
    claims: dict = {}
