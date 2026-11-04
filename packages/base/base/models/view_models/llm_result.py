"""What a model call returned, in terms no provider owns."""

from base.models.definitions.model import FinishReason
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredMessage, StoredToolCall


class LLMUsageCounts(BaseModel):
    """Tokens one call consumed, split the way providers bill them."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    """Share of `input_tokens` read from the provider's prompt cache."""

    cache_write_tokens: int = 0
    """Share of `input_tokens` written to it, charged at a premium."""

    reasoning_tokens: int = 0
    """Output tokens spent thinking rather than answering, where reported."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResult(BaseModel):
    """One completed model call."""

    content: str | None = None
    tool_calls: list[StoredToolCall] | None = None
    usage: LLMUsageCounts | None = None
    reasoning: str | None = None
    """Thinking the provider chose to return. Absent on most calls."""

    finish_reason: FinishReason | None = None
    """Why generation stopped, normalised. Branch on this rather than on
    anything provider-shaped: `LENGTH` means the reply is truncated."""


class ParsedResult[T: BaseModel](BaseModel):
    """A call whose output was parsed into the schema the caller asked for."""

    parsed: T
    usage: LLMUsageCounts | None = None


class ToolRunResult(BaseModel):
    """What a tool-calling turn produced."""

    result: LLMResult
    """The final assistant turn, the one with no further tool calls."""

    messages: list[StoredMessage]
    """Assistant and tool messages generated during the turn, in order.

    Returned rather than persisted: whether a turn belongs in a conversation
    is the caller's decision, not this runner's.
    """

    rounds: int
    """How many times the model was called. One means it answered directly."""
