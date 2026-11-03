from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from base.models.view_models import BaseModel


class CreateConversationRequest(BaseModel):
    title: str | None = "New Chat"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateConversationResponse(BaseModel):
    conversation_id: UUID


class GetConversationResponse(BaseModel):
    conversation_id: UUID
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class EditConversationRequest(BaseModel):
    title: str | None = None
    metadata: dict[str, Any] | None = None


class EditConversationResponse(BaseModel):
    pass


class DeleteConversationResponse(BaseModel):
    pass
