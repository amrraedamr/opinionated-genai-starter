from enum import StrEnum


class ProviderType(StrEnum):
    AZURE_OPENAI = "azure-openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    VLLM = "vllm"
    GROQ = "groq"
    TOGETHER = "together"
    LITELLM = "litellm"


class ReasoningEffort(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class FinishReason(StrEnum):
    """Why a model stopped generating, in terms every provider can express."""

    STOP = "stop"
    """Reached a natural end, or a stop sequence."""

    TOOL_CALLS = "tool_calls"
    """Wants a tool run before it can continue."""

    LENGTH = "length"
    """Hit the output token cap. The reply is truncated."""

    CONTENT_FILTER = "content_filter"
    """Blocked by the provider's safety layer."""

    OTHER = "other"
    """Something provider-specific the seam does not model."""
