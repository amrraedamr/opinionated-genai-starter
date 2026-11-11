"""The turn that runs a model to a final answer, calling tools on the way."""

from collections.abc import Sequence
from typing import Any, Protocol

from base.models.definitions.guardrails import GuardrailStage
from base.models.definitions.message import MessageRole
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import ConversationId, StoredMessage
from base.models.view_models.llm_result import LLMResult, ToolRunResult
from base.models.view_models.prompt import RenderedPrompt
from base.models.view_models.streams import (
    AnyStreamMessage,
    StreamEnd,
    StreamError,
    StreamStart,
)
from base.services.guardrails.factory import build_guardrail_pipeline
from base.services.guardrails.pipeline import GuardrailPipeline
from base.services.llm import AbstractLLMService
from base.services.llm.openai import OpenAIService
from base.services.logging.console import ConsoleLoggerService
from base.services.tool_calling.toolbox import Tool, Toolbox, ToolError
from base.tools.stream import StreamIterator


class Completion(Protocol):
    """One model call, as `AbstractLLMService.acomplete` or an agent's guarded `_acomplete` makes it."""

    async def __call__(
        self,
        *,
        prompt: RenderedPrompt | None = None,
        messages: Sequence[StoredMessage] | None = None,
        tools: Sequence[Tool[BaseModel]] | None = None,
        **kwargs: Any,
    ) -> LLMResult: ...


class _TurnStream(StreamIterator):
    """Passes every round's text into one stream, starting it once and leaving the end to the runner."""

    def __init__(self, inner: StreamIterator) -> None:
        super().__init__()
        self._inner = inner
        self._started = False

    async def __anext__(self) -> dict[str, str | int] | str:
        return await self._inner.__anext__()

    def send(self, value: AnyStreamMessage) -> None:
        if self._passes(value):
            self._inner.send(value)

    async def asend(self, value: AnyStreamMessage) -> None:
        if self._passes(value):
            await self._inner.asend(value)

    def end(self, end_stream_model: StreamEnd | None = None) -> None:
        del end_stream_model

    async def aend(self, end_stream_model: StreamEnd | None = None) -> None:
        del end_stream_model

    def error(self, error_stream_model: StreamError | None = None) -> None:
        self._inner.error(error_stream_model)

    async def aerror(self, error_stream_model: StreamError | None = None) -> None:
        await self._inner.aerror(error_stream_model)

    def _passes(self, value: AnyStreamMessage) -> bool:
        if isinstance(value, StreamEnd):
            return False
        if isinstance(value, StreamStart):
            if self._started:
                return False
            self._started = True
        return True


class ToolRunner:
    """Calls the model until it stops asking for tools, at most `max_rounds` times.

    Each round sends the conversation plus the toolbox, runs what the model asked
    for, appends the results, and asks again. Pass `complete` to run each round
    through an agent's own guarded call instead of the bare provider.
    """

    def __init__(
        self,
        toolbox: Toolbox,
        max_rounds: int = 5,
        llm_service: AbstractLLMService | None = None,
        guardrails: GuardrailPipeline | None = None,
        complete: Completion | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ToolError("max_rounds must be at least 1")
        self._toolbox = toolbox
        self._max_rounds = max_rounds
        self._complete: Completion = (
            complete or (llm_service or OpenAIService()).acomplete
        )
        self._logger: ConsoleLoggerService = ConsoleLoggerService()
        self._guardrails: GuardrailPipeline = (
            guardrails if guardrails is not None else build_guardrail_pipeline()
        )

    async def arun(
        self,
        messages: Sequence[StoredMessage] = (),
        *,
        prompt: RenderedPrompt | None = None,
        conversation_id: ConversationId | None = None,
        stream_iterator: StreamIterator | None = None,
        **kwargs: Any,
    ) -> ToolRunResult:
        """Run the turn to a final answer, from `messages`, a `prompt`, or both.

        `messages` is left untouched; everything the turn adds comes back in the
        result. A streamed turn ends its stream once, after the final round.
        Raises `ToolError` if the model is still asking for tools after `max_rounds`.
        """
        if not messages and prompt is None:
            raise ToolError(
                "A tool-calling turn needs a prompt or at least one message"
            )
        conversation_key = messages[0].conversation_id if messages else conversation_id
        if conversation_key is None:
            raise ToolError("A turn from a prompt alone needs its conversation_id")

        conversation = list(messages)
        produced: list[StoredMessage] = []
        sequence = max((message.sequence for message in messages), default=0)
        if stream_iterator is not None:
            kwargs["stream_iterator"] = _TurnStream(stream_iterator)

        for round_number in range(1, self._max_rounds + 1):
            result = await self._complete(
                prompt=prompt,
                messages=conversation or None,
                tools=self._toolbox.tools,
                **kwargs,
            )

            tool_calls = result.tool_calls
            if not tool_calls:
                if stream_iterator is not None:
                    await stream_iterator.asend(StreamEnd())
                return ToolRunResult(
                    result=result, messages=produced, rounds=round_number
                )

            sequence += 1
            assistant = StoredMessage(
                conversation_id=conversation_key,
                sequence=sequence,
                role=MessageRole.ASSISTANT,
                content=result.content,
                tool_calls=tool_calls,
            )
            conversation.append(assistant)
            produced.append(assistant)

            for call in tool_calls:
                self._logger.info(f"Tool call: {call.function.name}")
                sequence += 1
                outcome = StoredMessage(
                    conversation_id=conversation_key,
                    sequence=sequence,
                    role=MessageRole.TOOL,
                    content=await self._checked(await self._toolbox.ainvoke(call)),
                    tool_call_id=call.id,
                )
                conversation.append(outcome)
                produced.append(outcome)

        raise ToolError(
            f"Still calling tools after {self._max_rounds} rounds; "
            "raise max_rounds or narrow the toolbox."
        )

    async def _checked(self, content: str) -> str:
        """A tool result, checked before the model reads it; a refusal raises and ends the turn."""
        if not self._guardrails.applies_to(GuardrailStage.TOOL_RESULT):
            return content
        return await self._guardrails.aenforce(content, GuardrailStage.TOOL_RESULT)
