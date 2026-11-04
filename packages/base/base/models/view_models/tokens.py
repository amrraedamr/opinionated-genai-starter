"""Token types, shared by both halves of authentication.

They live outside `services/auth/` because importing that package initialises
it, and it guards on the `auth` extra. A service that only authenticates
itself signs and verifies nothing, so it should not need pyjwt to say so.
"""

from typing import NewType

from pydantic import BaseModel as PydanticBaseModel

from base.models.view_models import BaseModel

AccessToken = NewType("AccessToken", str)
RefreshToken = NewType("RefreshToken", str)


class TokenPair(BaseModel):
    access_token: AccessToken
    refresh_token: RefreshToken


class TokenResponse(PydanticBaseModel):
    """An OAuth2 token endpoint's reply, per RFC 6749 section 5.1.

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


__all__ = ["AccessToken", "RefreshToken", "TokenPair", "TokenResponse"]
