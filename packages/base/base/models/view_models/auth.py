"""Who a verified caller is, as the token claims describe them."""

from uuid import UUID

from base.models.view_models import BaseModel


class Principal(BaseModel):
    """The verified caller, reconstructed from token claims alone."""

    subject: str
    user_id: UUID | None = None
    email: str | None = None
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    issuer: str | None = None
    token_type: str | None = None
    claims: dict = {}
