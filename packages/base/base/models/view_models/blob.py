"""What an object store knows about a blob, apart from its bytes."""

from base.models.view_models import BaseModel


class BlobProperties(BaseModel):
    """One blob's metadata, in terms every object store can answer.

    Listing or inspecting blobs returns these rather than the store client's
    own type, so reading what a container holds does not name the vendor.
    """

    name: str = ""
    size: int = 0
    content_type: str = "application/octet-stream"
    etag: str | None = None
    last_modified: str | None = None

    metadata: dict[str, str] = {}
    """Whatever the caller stored alongside the blob."""
