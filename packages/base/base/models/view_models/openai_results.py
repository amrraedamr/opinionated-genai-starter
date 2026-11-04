"""OpenAI wire-shaped results. Internal to this adapter; callers get LLMResult."""

from base.tools.errors import MissingExtraError

try:
    from openai.types.chat import (
        ChatCompletionMessage,
        ChatCompletionMessageFunctionToolCall,
        ChatCompletionMessageParam,
    )
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses import (
        Response,
        ResponseFunctionToolCall,
        ResponseInputItemParam,
        ResponseInputParam,
        ResponseReasoningItem,
        ResponseUsage,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="OpenAI result models") from exc

from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredToolCall, StoredToolCallFunction
from base.services.logging.console import ConsoleLoggerService


class CompletionResult(BaseModel):
    """Raw chat completion reply, before the adapter neutralises it."""

    message: ChatCompletionMessage
    usage: CompletionUsage | None = None
    finish_reason: str | None = None

    @property
    def content(self) -> str | None:
        return self.message.content

    @property
    def tool_calls(self) -> list[StoredToolCall] | None:
        if not self.message.tool_calls:
            return None
        result: list[StoredToolCall] = []
        for tool_call in self.message.tool_calls:
            if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                ConsoleLoggerService().warning(
                    f"Dropping unsupported tool call type in CompletionResult: "
                    f"{type(tool_call).__name__}"
                )
                continue
            result.append(
                StoredToolCall(
                    id=tool_call.id,
                    function=StoredToolCallFunction(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
            )
        return result or None


class ResponseResult(BaseModel):
    """Raw Responses API reply, before the adapter neutralises it."""

    response: Response
    usage: ResponseUsage | None = None

    @property
    def content(self) -> str | None:
        return self.response.output_text or None

    @property
    def reasoning(self) -> str | None:
        """Reasoning summaries, when the call asked the model to return them."""
        summaries = [
            part.text
            for item in self.response.output
            if isinstance(item, ResponseReasoningItem)
            for part in item.summary
            if part.text
        ]
        return "\n".join(summaries) or None

    @property
    def tool_calls(self) -> list[StoredToolCall] | None:
        tool_calls = [
            StoredToolCall(
                id=item.call_id,
                function=StoredToolCallFunction(
                    name=item.name,
                    arguments=item.arguments,
                ),
            )
            for item in self.response.output
            if isinstance(item, ResponseFunctionToolCall)
        ]
        return tool_calls or None


class ParsedCompletionResult[T: BaseModel](BaseModel):
    """Result of a structured-output chat completion call."""

    parsed: T
    usage: CompletionUsage | None = None


class ParsedResponseResult[T: BaseModel](BaseModel):
    """Result of a structured-output Responses API call."""

    parsed: T
    usage: ResponseUsage | None = None


type LLMPromptItem = ChatCompletionMessageParam | ResponseInputItemParam
"""Element type of a captured prompt list."""

type LLMPrompt = list[ChatCompletionMessageParam] | ResponseInputParam
"""Input type at the call boundary: callers pass one of the two concrete
list types, which Pyright accepts without contortions."""

type LLMUsage = CompletionUsage | ResponseUsage | None
"""Token usage from either surface; `None` when the call errored first."""
