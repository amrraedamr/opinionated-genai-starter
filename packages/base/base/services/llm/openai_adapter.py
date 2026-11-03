from base.tools.errors import MissingExtraError

try:
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionDeveloperMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCallParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionUserMessageParam,
    )
    from openai.types.chat.chat_completion_message_function_tool_call_param import (
        Function as ChatCompletionMessageFunctionToolCallParam,
    )
    from openai.types.responses import (
        EasyInputMessageParam,
        ResponseInputItemParam,
        ResponseInputParam,
    )
    from openai.types.responses.response_function_tool_call_param import (
        ResponseFunctionToolCallParam,
    )
    from openai.types.responses.response_input_param import FunctionCallOutput
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="OpenAIMessageAdapter") from exc

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import StoredMessage
from base.models.view_models.prompt import (
    RenderedPrompt,
    RenderedPromptInstruction,
    RenderedPromptMessage,
)


class OpenAIMessageAdapter:
    # Chat Completions

    @classmethod
    def to_chat_completions_messages(
        cls,
        *,
        messages: list[StoredMessage] | None = None,
        prompt: RenderedPrompt | None = None,
    ) -> list[ChatCompletionMessageParam]:
        if messages is None and prompt is None:
            raise ValueError("At least one of `messages` or `prompt` must be provided")

        result: list[ChatCompletionMessageParam] = []

        if prompt is not None:
            for instruction in prompt.instructions:
                result.append(cls._to_chat_completions_instruction(instruction))
            for message in prompt.messages:
                result.append(cls._to_chat_completions_prompt_message(message))

        if messages is not None:
            for message in messages:
                result.append(cls._to_chat_completions_stored_message(message))

        return result

    @staticmethod
    def _to_chat_completions_instruction(
        instruction: RenderedPromptInstruction,
    ) -> ChatCompletionMessageParam:
        if instruction.role == MessageRole.SYSTEM:
            return ChatCompletionSystemMessageParam(
                role="system",
                content=instruction.content,
            )
        if instruction.role == MessageRole.DEVELOPER:
            return ChatCompletionDeveloperMessageParam(
                role="developer",
                content=instruction.content,
            )
        raise ValueError(
            f"Unsupported prompt instruction role for chat completions: {instruction.role}"
        )

    @staticmethod
    def _to_chat_completions_prompt_message(
        message: RenderedPromptMessage,
    ) -> ChatCompletionMessageParam:
        content = message.content or ""
        if message.role == MessageRole.USER:
            return ChatCompletionUserMessageParam(role="user", content=content)
        if message.role == MessageRole.ASSISTANT:
            return ChatCompletionAssistantMessageParam(
                role="assistant", content=content
            )
        if message.role == MessageRole.TOOL:
            if not message.tool_call_id:
                raise ValueError("Prompt TOOL message is missing tool_call_id")
            return ChatCompletionToolMessageParam(
                role="tool",
                content=content,
                tool_call_id=message.tool_call_id,
            )
        raise ValueError(
            f"Unsupported prompt message role for chat completions: {message.role}"
        )

    @staticmethod
    def _to_chat_completions_stored_message(
        message: StoredMessage,
    ) -> ChatCompletionMessageParam:
        if message.role == MessageRole.SYSTEM:
            return ChatCompletionSystemMessageParam(
                role="system",
                content=message.content or "",
            )
        if message.role == MessageRole.DEVELOPER:
            return ChatCompletionDeveloperMessageParam(
                role="developer",
                content=message.content or "",
            )
        if message.role == MessageRole.USER:
            return ChatCompletionUserMessageParam(
                role="user",
                content=message.content or "",
            )
        if message.role == MessageRole.ASSISTANT:
            param: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": message.content,
            }
            if message.tool_calls:
                param["tool_calls"] = [
                    ChatCompletionMessageToolCallParam(
                        id=tool_call.id,
                        type="function",
                        function=ChatCompletionMessageFunctionToolCallParam(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                    for tool_call in message.tool_calls
                ]
            return param
        if message.role == MessageRole.TOOL:
            if not message.tool_call_id:
                raise ValueError("Stored TOOL message is missing tool_call_id")
            return ChatCompletionToolMessageParam(
                role="tool",
                content=message.content or "",
                tool_call_id=message.tool_call_id,
            )
        raise ValueError(
            f"Unsupported stored message role for chat completions: {message.role}"
        )

    # Responses

    @classmethod
    def to_responses_input(
        cls,
        *,
        messages: list[StoredMessage] | None = None,
        prompt: RenderedPrompt | None = None,
    ) -> ResponseInputParam:
        if messages is None and prompt is None:
            raise ValueError("At least one of `messages` or `prompt` must be provided")

        items: list[ResponseInputItemParam] = []

        if prompt is not None:
            for instruction in prompt.instructions:
                items.append(cls._to_responses_instruction(instruction))
            for message in prompt.messages:
                items.extend(cls._to_responses_prompt_items(message))

        if messages is not None:
            for message in messages:
                items.extend(cls._to_responses_stored_items(message))

        return items

    @staticmethod
    def _to_responses_instruction(
        instruction: RenderedPromptInstruction,
    ) -> EasyInputMessageParam:
        if instruction.role == MessageRole.SYSTEM:
            return EasyInputMessageParam(role="system", content=instruction.content)
        if instruction.role == MessageRole.DEVELOPER:
            return EasyInputMessageParam(role="developer", content=instruction.content)
        raise ValueError(
            f"Unsupported prompt instruction role for responses: {instruction.role}"
        )

    @staticmethod
    def _to_responses_prompt_items(
        message: RenderedPromptMessage,
    ) -> list[ResponseInputItemParam]:
        content = message.content or ""
        if message.role == MessageRole.USER:
            return [EasyInputMessageParam(role="user", content=content)]
        if message.role == MessageRole.ASSISTANT:
            return [EasyInputMessageParam(role="assistant", content=content)]
        if message.role == MessageRole.SYSTEM:
            return [EasyInputMessageParam(role="system", content=content)]
        if message.role == MessageRole.DEVELOPER:
            return [EasyInputMessageParam(role="developer", content=content)]
        if message.role == MessageRole.TOOL:
            if not message.tool_call_id:
                raise ValueError("Prompt TOOL message is missing tool_call_id")
            return [
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=message.tool_call_id,
                    output=content,
                )
            ]
        raise ValueError(
            f"Unsupported prompt message role for responses: {message.role}"
        )

    @staticmethod
    def _to_responses_stored_items(
        message: StoredMessage,
    ) -> list[ResponseInputItemParam]:
        content = message.content or ""
        if message.role == MessageRole.USER:
            return [EasyInputMessageParam(role="user", content=content)]
        if message.role == MessageRole.SYSTEM:
            return [EasyInputMessageParam(role="system", content=content)]
        if message.role == MessageRole.DEVELOPER:
            return [EasyInputMessageParam(role="developer", content=content)]
        if message.role == MessageRole.ASSISTANT:
            items: list[ResponseInputItemParam] = []
            # Skip empty content — tool-only assistant turns shouldn't emit a text item
            if message.content:
                items.append(
                    EasyInputMessageParam(role="assistant", content=message.content)
                )
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    items.append(
                        ResponseFunctionToolCallParam(
                            type="function_call",
                            call_id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        )
                    )
            return items
        if message.role == MessageRole.TOOL:
            if not message.tool_call_id:
                raise ValueError("Stored TOOL message is missing tool_call_id")
            return [
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=message.tool_call_id,
                    output=content,
                )
            ]
        raise ValueError(
            f"Unsupported stored message role for responses: {message.role}"
        )
