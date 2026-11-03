import asyncio
import threading
from typing import Any

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import (
    ChatHistoryConfig,
    ConversationId,
    StoredMessage,
    StoredToolCall,
)
from base.services.common.meta import KeyedSingleton
from base.services.logging.console import ConsoleLoggerService
from base.services.store.chat_history import ChatHistoryStore
from base.services.store.chat_history.memory import InMemoryChatHistoryStore
from base.tools.tokenizer import TokenCounter


class ChatHistoryService(metaclass=KeyedSingleton):
    """Append and retrieve canonical conversation history.

    Composes a ChatHistoryStore and TokenCounter to provide chat history
    persistence on top of a backend store. The store is supplied as a *type*
    rather than an instance so this `KeyedSingleton`'s identity key stays
    hashable; a service passes its own (e.g. `SqlChatHistoryStore`), and the
    in-memory reference implementation is the default. The service owns canonical
    stored-message operations, visibility filtering, tool-pair repair,
    and token counting.

    Per-conversation locking serializes ``next_sequence + append`` for a
    conversation so sequence assignment never races the SQL unique
    constraint. Sync and async paths use independent locks; mixing sync
    and async callers on the same conversation is undefined and not
    supported.

    Summarization lives in a separate ChatContextReducer service.
    """

    def __init__(
        self,
        config: ChatHistoryConfig,
        store: type[ChatHistoryStore] = InMemoryChatHistoryStore,
    ) -> None:
        self._config: ChatHistoryConfig = config

        self._store: ChatHistoryStore = store()
        self._token_counter: TokenCounter = TokenCounter(
            model=self._config.tokenizer_model
        )
        self._logger: ConsoleLoggerService = ConsoleLoggerService()

        self._async_append_locks: dict[ConversationId, asyncio.Lock] = {}
        self._async_append_locks_guard: asyncio.Lock = asyncio.Lock()

        self._sync_append_locks: dict[ConversationId, threading.Lock] = {}
        self._sync_append_locks_guard: threading.Lock = threading.Lock()

    # Lock resolution

    async def _aappend_lock_for(self, conversation_id: ConversationId) -> asyncio.Lock:
        async with self._async_append_locks_guard:
            lock = self._async_append_locks.get(conversation_id)
            if lock is None:
                lock = asyncio.Lock()
                self._async_append_locks[conversation_id] = lock
            return lock

    def _append_lock_for(self, conversation_id: ConversationId) -> threading.Lock:
        with self._sync_append_locks_guard:
            lock = self._sync_append_locks.get(conversation_id)
            if lock is None:
                lock = threading.Lock()
                self._sync_append_locks[conversation_id] = lock
            return lock

    # Append (async)

    async def aappend_message(
        self,
        conversation_id: ConversationId,
        *,
        role: MessageRole,
        content: str | None = None,
        tool_calls: list[StoredToolCall] | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredMessage:
        """Append a canonical message to a conversation."""
        return await self._aappend(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            metadata=metadata,
        )

    async def _aappend(
        self,
        *,
        conversation_id: ConversationId,
        role: MessageRole,
        content: str | None = None,
        tool_calls: list[StoredToolCall] | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredMessage:
        lock = await self._aappend_lock_for(conversation_id)
        async with lock:
            sequence = await self._store.anext_sequence(conversation_id)
            draft = self._build_draft(
                conversation_id=conversation_id,
                sequence=sequence,
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                metadata=metadata,
            )
            persisted = await self._store.aappend(conversation_id, [draft])
        return persisted[0]

    # Append (sync)

    def append_message(
        self,
        conversation_id: ConversationId,
        *,
        role: MessageRole,
        content: str | None = None,
        tool_calls: list[StoredToolCall] | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredMessage:
        """Append a canonical message to a conversation."""
        return self._append(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            metadata=metadata,
        )

    def _append(
        self,
        *,
        conversation_id: ConversationId,
        role: MessageRole,
        content: str | None = None,
        tool_calls: list[StoredToolCall] | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredMessage:
        lock = self._append_lock_for(conversation_id)
        with lock:
            sequence = self._store.next_sequence(conversation_id)
            draft = self._build_draft(
                conversation_id=conversation_id,
                sequence=sequence,
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                metadata=metadata,
            )
            persisted = self._store.append(conversation_id, [draft])
        return persisted[0]

    # Draft construction

    def _build_draft(
        self,
        *,
        conversation_id: ConversationId,
        sequence: int,
        role: MessageRole,
        content: str | None,
        tool_calls: list[StoredToolCall] | None,
        tool_call_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> StoredMessage:
        token_count = self._count_message_tokens(content, tool_calls)
        return StoredMessage(
            id=None,
            conversation_id=conversation_id,
            sequence=sequence,
            role=MessageRole(role),
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            token_count=token_count,
            metadata=metadata or {},
        )

    def _count_message_tokens(
        self,
        content: str | None,
        tool_calls: list[StoredToolCall] | None,
    ) -> int:
        total = self._token_counter.count(content or "")
        if tool_calls:
            for tool_call in tool_calls:
                total += self._token_counter.count(tool_call.function.name)
                total += self._token_counter.count(tool_call.function.arguments)
        return total

    # Fetch (async)

    async def aget_messages(
        self,
        conversation_id: ConversationId,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        """Return canonical messages for a conversation.

        If ``limit`` is set, returns the most recent ``limit`` messages
        (ascending order), after visibility filtering and tool-pair repair.
        """
        return await self._afetch_filtered(conversation_id, limit=limit)

    # Fetch (sync)

    def get_messages(
        self,
        conversation_id: ConversationId,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        """Return canonical messages for a conversation.

        If ``limit`` is set, returns the most recent ``limit`` messages
        (ascending order), after visibility filtering and tool-pair repair.
        """
        return self._fetch_filtered(conversation_id, limit=limit)

    # Management

    async def aclear(self, conversation_id: ConversationId) -> None:
        """Clear all messages for a conversation."""
        await self._store.aclear(conversation_id)

    def clear(self, conversation_id: ConversationId) -> None:
        """Clear all messages for a conversation."""
        self._store.clear(conversation_id)

    # Filtering

    async def _afetch_filtered(
        self,
        conversation_id: ConversationId,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        all_messages = await self._store.aget(conversation_id, limit=limit)
        visible = self._filter_visible(all_messages)
        return self._repair_tool_pairs(visible)

    def _fetch_filtered(
        self,
        conversation_id: ConversationId,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        all_messages = self._store.get(conversation_id, limit=limit)
        visible = self._filter_visible(all_messages)
        return self._repair_tool_pairs(visible)

    @staticmethod
    def _filter_visible(messages: list[StoredMessage]) -> list[StoredMessage]:
        return [
            message
            for message in messages
            if not message.metadata.get("is_policy_violated", False)
        ]

    @staticmethod
    def _repair_tool_pairs(
        messages: list[StoredMessage],
    ) -> list[StoredMessage]:
        """Ensure tool-call contract holds for LLM submission.

        Rules:
          - Every tool message must follow an assistant with a matching
            tool_call_id in its tool_calls. Orphan tools are dropped.
          - Every assistant with tool_calls must have tool results for
            every tool_call_id. Incomplete assistants are demoted to
            plain assistants (tool_calls stripped).
        """
        result: list[StoredMessage] = []
        index = 0
        while index < len(messages):
            message = messages[index]

            if message.role == MessageRole.TOOL:
                if (
                    result
                    and result[-1].role == MessageRole.ASSISTANT
                    and result[-1].tool_calls
                    and any(
                        tool_call.id == message.tool_call_id
                        for tool_call in result[-1].tool_calls
                    )
                ):
                    result.append(message)
                index += 1
                continue

            if message.role == MessageRole.ASSISTANT and message.tool_calls:
                expected_ids = {tool_call.id for tool_call in message.tool_calls}
                lookahead = index + 1
                found_ids: set[str] = set()
                while (
                    lookahead < len(messages)
                    and messages[lookahead].role == MessageRole.TOOL
                    and messages[lookahead].tool_call_id in expected_ids
                ):
                    found_ids.add(messages[lookahead].tool_call_id or "")
                    lookahead += 1

                if found_ids == expected_ids:
                    result.append(message)
                    for tool_index in range(index + 1, lookahead):
                        result.append(messages[tool_index])
                    index = lookahead
                else:
                    demoted = message.model_copy(update={"tool_calls": None})
                    result.append(demoted)
                    index = lookahead
                continue

            result.append(message)
            index += 1

        return result
