from typing import Any

from pydantic import Field

from base.models.view_models import BaseModel


class SendChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=16_000)


class SendChatResponse(BaseModel):
    content: str


class ChatMessageView(BaseModel):
    role: str
    content: str | None = None
    sequence: int
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ListChatResponse(BaseModel):
    messages: list[ChatMessageView]
