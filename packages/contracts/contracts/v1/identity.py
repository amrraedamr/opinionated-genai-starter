from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

USER_REGISTERED = "identity.user.registered"
USER_DEACTIVATED = "identity.user.deactivated"


class UserProfile(BaseModel):
    """A user as other services may see them."""

    user_id: UUID
    email: str
    first_name: str | None = None
    last_name: str | None = None


class UserRegistered(BaseModel):
    user_id: UUID
    email: str
    registered_at: datetime


class UserDeactivated(BaseModel):
    user_id: UUID
    deactivated_at: datetime
