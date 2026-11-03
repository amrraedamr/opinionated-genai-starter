"""The turn that runs a model to a final answer, calling tools on the way."""

from typing import Any, Literal

from base.models.definitions.message import MessageRole
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredMessage
from base.models.view_models.llm_result import CompletionResult, ResponseResult
from base.services.llm import OpenAIService
from base.services.llm.openai_adapter import OpenAIMessageAdapter
from base.services.logging.console import ConsoleLoggerService
from base.services.tool_calling.toolbox import Toolbox, ToolError

LLMApi = Literal["completions", "responses"]


class ToolRunResult(BaseModel):
    """What a tool-calling turn produced."""

    result: CompletionResult | ResponseResult
    """The final assistant turn, the one with no further tool calls."""

    messages: list[StoredMessage]
    """Assistant and tool messages generated during the turn, in order.

    Returned rather than persisted: whether a turn belongs in a conversation
    is the caller's decision, not this runner's.
    """

    rounds: int
    """How many times the model was called. One means it answered directly."""


class ToolRunner:
    """Calls the model until it stops asking for tools.

    Each round sends the conversation plus the toolbox, runs whatever the
    model asked for, appends the results, and asks again. `max_rounds` is the
    stop: a model that keeps calling tools would otherwise bill forever.

    Both OpenAI surfaces are supported, because the two differ only in how
    the request is shaped — the loop around them is the same.
    """

    def __init__(
        self,
        toolbox: Toolbox,
        max_rounds: int = 5,
        api: LLMApi = "completions",
    ) -> None:
        if max_rounds < 1:
            raise ToolError("max_rounds must be at least 1")
        self._toolbox = toolbox
        self._max_rounds = max_rounds
        self._api: LLMApi = api
        self._llm_service: OpenAIService = OpenAIService()
        self._logger: ConsoleLoggerService = ConsoleLoggerService()

    async def arun(
        self, messages: list[StoredMessage], **kwargs: Any
    ) -> ToolRunResult:
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
            result = await self._acall(conversation, **kwargs)

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
                    content=await self._toolbox.ainvoke(call),
                    tool_call_id=call.id,
                )
                conversation.append(outcome)
                produced.append(outcome)

        raise ToolError(
            f"Still calling tools after {self._max_rounds} rounds; "
            "raise max_rounds or narrow the toolbox."
        )

    async def _acall(
        self, conversation: list[StoredMessage], **kwargs: Any
    ) -> CompletionResult | ResponseResult:
        if self._api == "responses":
            return await self._llm_service.aresponses(
                OpenAIMessageAdapter.to_responses_input(messages=conversation),
                tools=self._toolbox.responses_schemas,
                **kwargs,
            )
        return await self._llm_service.acompletions(
            OpenAIMessageAdapter.to_chat_completions_messages(messages=conversation),
            tools=self._toolbox.schemas,
            **kwargs,
        )
