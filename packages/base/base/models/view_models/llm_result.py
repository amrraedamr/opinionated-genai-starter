from base.tools.errors import MissingExtraError

try:
    from openai.types.chat import (
        ChatCompletionMessage,
        ChatCompletionMessageFunctionToolCall,
    )
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses import Response, ResponseFunctionToolCall, ResponseUsage
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="LLM result models") from exc

from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredToolCall, StoredToolCallFunction
from base.services.logging.console import ConsoleLoggerService


class CompletionResult(BaseModel):
    """Result of a chat completion call (streaming or non-streaming)."""

    message: ChatCompletionMessage
    usage: CompletionUsage | None = None

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
    """Result of a Responses API call (streaming or non-streaming)."""

    response: Response
    usage: ResponseUsage | None = None

    @property
    def content(self) -> str | None:
        return self.response.output_text or None

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
