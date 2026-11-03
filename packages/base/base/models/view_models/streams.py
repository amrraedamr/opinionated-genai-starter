from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import Field

from base.models.view_models import BaseModel

T = TypeVar("T")


class StreamModel(BaseModel):
    id: str | None = None
    card_type: str | None = None


class StreamStart(StreamModel):
    message_type: Literal["start"] = "start"


class StreamMessage(StreamModel, Generic[T]):
    data: T
    message_type: Literal["stream"] = "stream"


class StreamProgress(StreamModel):
    percentage: float | None = None
    label: str | None = None
    message_type: Literal["progress"] = "progress"


class StreamEnd(StreamModel):
    message_type: Literal["end"] = "end"


class StreamError(StreamModel):
    """Blocking stream message. Stops the iterator."""

    error_code: int | None = None
    text: str | None = None
    detail: dict[str, Any] | None = None
    message_type: Literal["error"] = "error"


AnyStreamMessage = Annotated[
    StreamStart | StreamMessage[Any] | StreamProgress | StreamEnd | StreamError,
    Field(discriminator="message_type"),
]
