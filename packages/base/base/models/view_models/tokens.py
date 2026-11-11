"""Token types, shared by both halves of authentication.

They live outside `services/auth/` because importing that package initialises
it, and it guards on the `auth` extra. A service that only authenticates
itself signs and verifies nothing, so it should not need pyjwt to say so.
"""

from datetime import UTC, datetime
from typing import NewType

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from base.models.view_models import BaseModel

AccessToken = NewType("AccessToken", str)
RefreshToken = NewType("RefreshToken", str)


class TokenPair(BaseModel):
    access_token: AccessToken
    refresh_token: RefreshToken


class ForwardedBearer(BaseModel):
    """The token a caller presented, kept to act as them downstream until it expires."""

    token: AccessToken = Field(repr=False)
    expires_at: datetime | None = None

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and datetime.now(UTC) >= self.expires_at


class TokenResponse(PydanticBaseModel):
    """An OAuth2 token endpoint's reply, per RFC 6749 section 5.1.

    Plain pydantic on purpose: the field names are the wire format a provider
    sends, so the shared base's camelised aliases would stop it parsing.

    Parsed by both the inbound authorization-code exchange and the outbound
    client-credentials grant. `expires_in` is optional because providers are
    inconsistent about sending it; `id_token` is absent from a pure
    client-credentials response.
    """

    access_token: AccessToken
    token_type: str = "Bearer"
    expires_in: int | None = None
    refresh_token: RefreshToken | None = None
    id_token: str | None = None
    scope: str | None = None


__all__ = [
    "AccessToken",
    "ForwardedBearer",
    "RefreshToken",
    "TokenPair",
    "TokenResponse",
]
