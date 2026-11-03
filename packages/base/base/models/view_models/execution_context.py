from base.models.definitions.database import Locale
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import ConversationId


class FlowVariables(BaseModel):
    """Single source of truth for one pipeline run."""

    # Conversation-level variables
    conversation_id: ConversationId | None = None

    # Inputs
    user_query: str = ""
    locale: Locale = Locale.EN

    # Outputs
    final_answer: str = ""
