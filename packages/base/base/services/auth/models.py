"""Token and identity types.

The token types live in `base.models.view_models.tokens` and are re-exported
here, so `base.clients.credentials` can reach them without initialising this
package — which guards on the `auth` extra.
"""

from uuid import UUID

from pydantic import BaseModel as PydanticBaseModel

from base.models.view_models.tokens import (
    AccessToken,
    RefreshToken,
    TokenPair,
    TokenResponse,
)


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


__all__ = [
    "AccessToken",
    "Principal",
    "RefreshToken",
    "TokenPair",
    "TokenResponse",
]
