from typing import Any, Literal, NewType, Self
from uuid import UUID

from pydantic import Field, model_validator

from base.models.definitions.message import MessageRole
from base.models.view_models import BaseModel
from base.models.view_models.models import OpenAIModels

# Identifiers

ConversationId = NewType("ConversationId", UUID)


# Tool calls


class StoredToolCallFunction(BaseModel):
    """Function invocation nested inside a tool call. Matches OpenAI's shape."""

    name: str
    arguments: str
    """JSON-encoded string, per OpenAI spec. Never a dict."""


class StoredToolCall(BaseModel):
    """A tool call issued by an assistant message.

    Currently only represents function-type tool calls. Custom tool calls
    from OpenAI's newer tool types are not supported — ChatHistoryService
    raises NotImplementedError on them at append time. If custom tool call
    support is needed later, expand this to a discriminated union.
    """

    id: str
    type: Literal["function"] = "function"
    function: StoredToolCallFunction


# Message


class StoredMessage(BaseModel):
    """Canonical storage shape for a single message in a conversation.

    Backend-neutral: SQL and Mongo stores both round-trip through this type.
    Satisfies the _HasSequence protocol via the `sequence` field.

    Persistence timestamps (created_at, updated_at) are owned by the storage
    layer, not the view model. Fetch them from the DTO/document when needed.

    Reserved metadata keys:
        is_policy_violated — bool, filtered out on fetch by ChatHistoryService
    """

    id: UUID | None = None
    """Populated by the store on append. None on freshly-constructed
    drafts, non-None on any message read from or returned by a store."""

    conversation_id: ConversationId

    sequence: int
    """Monotonic per-conversation order. Assigned at append time via
    store.anext_sequence."""

    role: MessageRole
    """The role of the message's author. Tool messages have role 'tool' and
    are linked to their parent assistant message via tool_call_id."""

    content: str | None = None
    """Null only for pure tool-call assistant messages."""

    tool_calls: list[StoredToolCall] | None = None
    """Assistant-role tool invocations. Null otherwise."""

    tool_call_id: str | None = None
    """Set on role=='tool' messages to pair with the originating assistant's
    tool_calls entry."""

    token_count: int | None = None
    """Cached count under the configured tokenizer. Computed at append time."""

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.role == MessageRole.SYSTEM:
            if self.content is None:
                raise ValueError("System messages require content.")

        elif self.role == MessageRole.DEVELOPER:
            if self.content is None:
                raise ValueError("Developer messages require content.")

        elif self.role == MessageRole.USER:
            if self.content is None:
                raise ValueError("User messages require content.")

        elif self.role == MessageRole.ASSISTANT:
            if self.content is None and not self.tool_calls:
                raise ValueError(
                    "Assistant messages require content, tool_calls, or both."
                )
        elif self.tool_calls is not None and self.role != MessageRole.ASSISTANT:
            raise ValueError("Only assistant messages may define tool_calls.")

        elif self.role == MessageRole.TOOL:
            if self.tool_call_id is None:
                raise ValueError("Tool messages require tool_call_id.")
        elif self.tool_call_id is not None and self.role != MessageRole.TOOL:
            raise ValueError("Only tool messages may define tool_call_id.")

        else:
            raise ValueError(f"Unsupported message role: {self.role}")

        return self


# Service configuration


class ChatHistoryConfig(BaseModel):
    """Configuration for ChatHistoryService."""

    tokenizer_model: OpenAIModels = OpenAIModels.GPT_4O
    """Passed to tiktoken.encoding_for_model(). Falls back to cl100k_base
    for unknown models."""
