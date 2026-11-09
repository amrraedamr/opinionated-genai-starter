"""The turn that runs a model to a final answer, calling tools on the way."""

from typing import Any

from base.models.definitions.guardrails import GuardrailStage
from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import StoredMessage
from base.models.view_models.llm_result import ToolRunResult
from base.services.guardrails.factory import build_guardrail_pipeline
from base.services.guardrails.pipeline import GuardrailPipeline
from base.services.llm import AbstractLLMService
from base.services.llm.openai import OpenAIService
from base.services.logging.console import ConsoleLoggerService
from base.services.tool_calling.toolbox import Toolbox, ToolError


class ToolRunner:
    """Calls the model until it stops asking for tools.

    Each round sends the conversation plus the toolbox, runs whatever the
    model asked for, appends the results, and asks again. `max_rounds` is the
    stop: a model that keeps calling tools would otherwise bill forever.

    The loop is provider-agnostic: it asks `AbstractLLMService` for a turn and
    reads tool calls off the neutral result.
    """

    def __init__(
        self,
        toolbox: Toolbox,
        max_rounds: int = 5,
        llm_service: AbstractLLMService | None = None,
        guardrails: GuardrailPipeline | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ToolError("max_rounds must be at least 1")
        self._toolbox = toolbox
        self._max_rounds = max_rounds
        self._llm_service: AbstractLLMService = llm_service or OpenAIService()
        self._logger: ConsoleLoggerService = ConsoleLoggerService()
        self._guardrails: GuardrailPipeline = (
            guardrails if guardrails is not None else build_guardrail_pipeline()
        )

    async def arun(self, messages: list[StoredMessage], **kwargs: Any) -> ToolRunResult:
        """Run the turn to a final answer.

        `messages` is left untouched; everything the turn adds comes back in
        the result. Raises `ToolError` if the model is still asking for tools
        after `max_rounds`.
        """
        if not messages:
            raise ToolError("A tool-calling turn needs at least one message")

        conversation = list(messages)
        produced: list[StoredMessage] = []
        conversation_id = messages[0].conversation_id
        sequence = max(message.sequence for message in messages)

        for round_number in range(1, self._max_rounds + 1):
            result = await self._llm_service.acomplete(
                messages=conversation, tools=self._toolbox.tools, **kwargs
            )

            tool_calls = result.tool_calls
            if not tool_calls:
                return ToolRunResult(
                    result=result, messages=produced, rounds=round_number
                )

            sequence += 1
            assistant = StoredMessage(
                conversation_id=conversation_id,
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
                    conversation_id=conversation_id,
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
        """A tool result, checked before it becomes something the model reads.

        This is the whole reason third-party tools are safe to enable. The
        result of an MCP server nobody here reviewed arrives as text the model
        was told to trust, so it is checked on the way in rather than only on
        the way back out to a caller.

        A refusal raises, which ends the turn. Substituting an apology would
        leave the model to continue on a result it never received.
        """
        if not self._guardrails.applies_to(GuardrailStage.TOOL_RESULT):
            return content
        return await self._guardrails.aenforce(content, GuardrailStage.TOOL_RESULT)
