from enum import StrEnum


class DatabaseTable(StrEnum):
    """Tables owned by chat_api."""

    PROMPT = "prompt"
    ACTIVE_PROMPT = "active_prompt"
    CONVERSATION = "conversation"
    CHAT_HISTORY_MESSAGE = "chat_history_message"
