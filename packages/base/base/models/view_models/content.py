"""Message content that is not just text."""

from typing import Annotated, Literal

from pydantic import Field

from base.models.view_models import BaseModel


class TextPart(BaseModel):
    """A run of plain text."""

    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An image, either by URL or inline base64."""

    type: Literal["image"] = "image"
    url: str | None = None
    data: str | None = None
    """Base64-encoded bytes. Requires `media_type`."""

    media_type: str | None = None
    detail: Literal["auto", "low", "high"] = "auto"
    """How much resolution the model should spend on it, where supported."""

    def model_post_init(self, _context: object) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("An ImagePart needs exactly one of `url` or `data`")
        if self.data is not None and self.media_type is None:
            raise ValueError("Inline image data needs a `media_type`")


class DocumentPart(BaseModel):
    """A file handed to the model whole, such as a PDF."""

    type: Literal["document"] = "document"
    file_id: str | None = None
    """Identifier of a file already uploaded to the provider."""

    data: str | None = None
    """Base64-encoded bytes. Requires `media_type` and `filename`."""

    media_type: str | None = None
    filename: str | None = None

    def model_post_init(self, _context: object) -> None:
        if (self.file_id is None) == (self.data is None):
            raise ValueError("A DocumentPart needs exactly one of `file_id` or `data`")
        if self.data is not None and not (self.media_type and self.filename):
            raise ValueError("Inline document data needs `media_type` and `filename`")


type ContentPart = Annotated[
    TextPart | ImagePart | DocumentPart, Field(discriminator="type")
]
