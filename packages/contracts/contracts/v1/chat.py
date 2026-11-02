from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

CONVERSATION_CREATED = "chat.conversation.created"


class ConversationCreated(BaseModel):
    conversation_id: UUID
    user_id: UUID
    created_at: datetime
