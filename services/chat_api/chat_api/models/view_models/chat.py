from typing import Any

from pydantic import Field

from base.models.view_models import BaseModel
from base.models.view_models.debug_trace import DebugTrace


class SendChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=16_000)


class SendChatResponse(BaseModel):
    content: str

    debug: DebugTrace | None = None
    """Present only when the caller asked for `?debug=true`. Carries the
    prompts and responses the models saw, so it is off by default."""


class ChatMessageView(BaseModel):
    role: str
    content: str | None = None
    sequence: int
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ListChatResponse(BaseModel):
    messages: list[ChatMessageView]
