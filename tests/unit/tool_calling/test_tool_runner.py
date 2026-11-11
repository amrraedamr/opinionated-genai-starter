"""`ToolRunner` drives a turn to a final answer and bills a bounded number of calls.

The loop is provider-agnostic, so a scripted `AbstractLLMService` stand-in is
enough to pin all of it: when it stops, what it appends, what it leaves the
caller's list alone, and that `max_rounds` is a real ceiling rather than a hint.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import (
    ConversationId,
    StoredMessage,
    StoredToolCall,
    StoredToolCallFunction,
)
from base.models.view_models.llm_result import LLMResult
from base.models.view_models.prompt import (
    RenderedPrompt,
    RenderedPromptInstruction,
    RenderedPromptMessage,
)
from base.models.view_models.streams import (
    StreamEnd,
    StreamError,
    StreamMessage,
    StreamStart,
)
from base.services.tool_calling import Tool, Toolbox, ToolError
from base.services.tool_calling.runner import ToolRunner
from base.tools.stream import StreamIterator

CONVERSATION = ConversationId(uuid.UUID(int=7))


class Args(BaseModel):
    city: str = "Doha"


class ScriptedLLM:
    """Answers with a canned `LLMResult` per round, repeating the last one."""

    def __init__(self, *replies: LLMResult) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def acomplete(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[index]


def call(identifier: str, name: str = "get_weather") -> StoredToolCall:
    return StoredToolCall(
        id=identifier,
        function=StoredToolCallFunction(name=name, arguments='{"city": "Doha"}'),
    )


def toolbox(result: str = "21") -> Toolbox:
    async def handler(args: Args) -> str:
        return f"{result} in {args.city}"

    return Toolbox(
        Tool(name="get_weather", description="d", parameters=Args, handler=handler)
    )


def user(content: str = "weather?", sequence: int = 1) -> StoredMessage:
    return StoredMessage(
        conversation_id=CONVERSATION,
        sequence=sequence,
        role=MessageRole.USER,
        content=content,
    )


def runner(llm: ScriptedLLM, max_rounds: int = 5) -> ToolRunner:
    return ToolRunner(toolbox(), max_rounds=max_rounds, llm_service=llm)  # type: ignore[arg-type]


async def test_a_reply_with_no_tool_calls_ends_the_turn_in_one_round() -> None:
    llm = ScriptedLLM(LLMResult(content="It is 21."))

    outcome = await runner(llm).arun([user()])

    assert outcome.rounds == 1
    assert outcome.result.content == "It is 21."
    assert outcome.messages == []
    assert len(llm.calls) == 1


async def test_a_tool_call_is_run_and_the_model_is_asked_again() -> None:
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1")]),
        LLMResult(content="It is 21 in Doha."),
    )

    outcome = await runner(llm).arun([user()])

    assert outcome.rounds == 2
    assert outcome.result.content == "It is 21 in Doha."
    assert [m.role for m in outcome.messages] == [
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert outcome.messages[1].content == "21 in Doha"
    assert outcome.messages[1].tool_call_id == "c1"


async def test_the_tool_results_are_fed_back_into_the_next_request() -> None:
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1")]),
        LLMResult(content="done"),
    )

    await runner(llm).arun([user()])

    second_request = llm.calls[1]["messages"]
    assert [m.role for m in second_request] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]


async def test_the_toolbox_is_offered_on_every_round() -> None:
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1")]),
        LLMResult(content="done"),
    )

    await runner(llm).arun([user()])

    assert all(request["tools"] for request in llm.calls)
    assert [t.name for t in llm.calls[0]["tools"]] == ["get_weather"]


async def test_several_tool_calls_in_one_reply_are_all_run_in_order() -> None:
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1"), call("c2"), call("c3")]),
        LLMResult(content="done"),
    )

    outcome = await runner(llm).arun([user()])

    assert [m.role for m in outcome.messages] == [
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
        MessageRole.TOOL,
    ]
    assert [m.tool_call_id for m in outcome.messages[1:]] == ["c1", "c2", "c3"]


async def test_produced_messages_continue_the_conversation_sequence() -> None:
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1"), call("c2")]),
        LLMResult(content="done"),
    )

    outcome = await runner(llm).arun([user(sequence=4)])

    assert [m.sequence for m in outcome.messages] == [5, 6, 7]
    assert all(m.conversation_id == CONVERSATION for m in outcome.messages)


async def test_the_callers_message_list_is_left_untouched() -> None:
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1")]),
        LLMResult(content="done"),
    )
    messages = [user()]

    await runner(llm).arun(messages)

    assert messages == [user()]


async def test_a_handler_that_raises_is_reported_to_the_model_not_the_caller() -> None:
    async def explodes(args: Args) -> str:
        raise RuntimeError("upstream down")

    box = Toolbox(
        Tool(name="get_weather", description="d", parameters=Args, handler=explodes)
    )
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1")]),
        LLMResult(content="sorry, the weather service is down"),
    )

    outcome = await ToolRunner(box, llm_service=llm).arun([user()])  # type: ignore[arg-type]

    assert "RuntimeError: upstream down" in (outcome.messages[1].content or "")
    assert outcome.result.content == "sorry, the weather service is down"


async def test_a_model_that_never_stops_calling_tools_is_cut_off() -> None:
    llm = ScriptedLLM(LLMResult(content=None, tool_calls=[call("c1")]))

    with pytest.raises(ToolError, match="Still calling tools after 3 rounds"):
        await runner(llm, max_rounds=3).arun([user()])

    assert len(llm.calls) == 3


async def test_one_round_is_a_legal_ceiling() -> None:
    llm = ScriptedLLM(LLMResult(content="straight answer"))

    outcome = await runner(llm, max_rounds=1).arun([user()])

    assert outcome.rounds == 1


def test_a_ceiling_below_one_is_refused_at_construction() -> None:
    with pytest.raises(ToolError, match="max_rounds must be at least 1"):
        ToolRunner(toolbox(), max_rounds=0, llm_service=ScriptedLLM())  # type: ignore[arg-type]


async def test_a_turn_with_no_messages_is_refused() -> None:
    llm = ScriptedLLM(LLMResult(content="x"))

    with pytest.raises(ToolError, match="a prompt or at least one message"):
        await runner(llm).arun([])


async def test_extra_keywords_are_forwarded_to_the_provider_on_every_round() -> None:
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1")]),
        LLMResult(content="done"),
    )

    await runner(llm).arun([user()], max_output_tokens=64, label="Runner")

    assert all(c["max_output_tokens"] == 64 for c in llm.calls)
    assert all(c["label"] == "Runner" for c in llm.calls)


# From a prompt


def prompt() -> RenderedPrompt:
    return RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content="be brief")
        ],
        messages=[RenderedPromptMessage(role=MessageRole.USER, content="weather?")],
    )


async def test_a_prompt_is_sent_on_every_round_with_the_tool_turns_after_it() -> None:
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1")]),
        LLMResult(content="done"),
    )

    outcome = await runner(llm).arun(prompt=prompt(), conversation_id=CONVERSATION)

    assert outcome.result.content == "done"
    assert all(c["prompt"] == prompt() for c in llm.calls)
    assert llm.calls[0]["messages"] is None
    assert [m.role for m in llm.calls[1]["messages"]] == [
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert [m.sequence for m in outcome.messages] == [1, 2]


async def test_a_turn_from_a_prompt_alone_needs_its_conversation() -> None:
    with pytest.raises(ToolError, match="conversation_id"):
        await runner(ScriptedLLM(LLMResult(content="x"))).arun(prompt=prompt())


async def test_each_round_goes_through_the_completion_it_was_given() -> None:
    """How an agent keeps its own guardrails on every round."""
    seen: list[dict[str, Any]] = []
    llm = ScriptedLLM(
        LLMResult(content=None, tool_calls=[call("c1")]),
        LLMResult(content="done"),
    )

    async def guarded(**kwargs: Any) -> LLMResult:
        seen.append(kwargs)
        return await llm.acomplete(**kwargs)

    await ToolRunner(toolbox(), complete=guarded).arun([user()])

    assert len(seen) == 2


# Streaming


class StreamingLLM(ScriptedLLM):
    """Streams each reply as a provider does: start, its text, end."""

    async def acomplete(self, **kwargs: Any) -> LLMResult:
        result = await super().acomplete(**kwargs)
        stream: StreamIterator = kwargs["stream_iterator"]
        await stream.asend(StreamStart())
        if result.content:
            await stream.asend(StreamMessage(data=result.content))
        await stream.asend(StreamEnd())
        return result


async def drained(stream: StreamIterator) -> list[str]:
    return [json.loads(str(frame["data"]))["message_type"] async for frame in stream]  # type: ignore[index]


async def test_a_streamed_turn_ends_once_after_its_last_round() -> None:
    """A round's end would otherwise close the stream before the answer arrives."""
    llm = StreamingLLM(
        LLMResult(content="checking", tool_calls=[call("c1")]),
        LLMResult(content="21 in Doha"),
    )
    stream = StreamIterator()

    await runner(llm).arun([user()], stream_iterator=stream)

    assert await drained(stream) == ["start", "stream", "stream", "end"]


async def test_a_streamed_error_still_reaches_the_caller() -> None:
    llm = ScriptedLLM(LLMResult(content=None, tool_calls=[call("c1")]))
    stream = StreamIterator()

    async def failing(**kwargs: Any) -> LLMResult:
        await kwargs["stream_iterator"].asend(StreamError(error_code=500, text="x"))
        return await llm.acomplete(**kwargs)

    with pytest.raises(ToolError):
        await ToolRunner(toolbox(), max_rounds=1, complete=failing).arun(
            [user()], stream_iterator=stream
        )

    assert await drained(stream) == ["error"]
