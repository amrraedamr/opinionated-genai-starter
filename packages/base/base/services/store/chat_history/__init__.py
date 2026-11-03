from abc import ABC

from base.models.view_models.chat_history import ConversationId, StoredMessage
from base.services.store import AbstractSequenceStore


class ChatHistoryStore(
    AbstractSequenceStore[ConversationId, StoredMessage],
    ABC,
):
    """Named contract for chat history stores.

    Marker subclass of AbstractSequenceStore parameterized for chat history.
    Adds no new methods — exists so concrete stores inherit from a grep-able,
    semantically-named type rather than a generic alias. ChatHistoryService
    depends on this type, not on AbstractSequenceStore directly."""
