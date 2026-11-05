"""What a vault knows about a secret, apart from its value."""

from datetime import datetime

from base.models.view_models import BaseModel


class SecretInfo(BaseModel):
    """One secret's metadata, in terms every vault can answer.

    Listing secrets returns these rather than the vault client's own type, so
    reading what a vault holds does not require naming the vendor.
    """

    name: str

    version: str | None = None
    """The version a vault assigned. None where versioning is not kept."""

    enabled: bool = True
    """Whether the vault will currently serve the value."""

    created_at: datetime | None = None
    updated_at: datetime | None = None
