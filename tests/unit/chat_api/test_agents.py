"""`TestAgent` — the one agent this service ships, and the shape a copy takes.

Most of what it is lives in its ClassVars, and one of them is load-bearing:
`ChatHistoryService` is keyed by the store *type*, so an agent naming a
different store than the routers would read a different conversation and
answer from an empty history.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import (
    ChatHistoryConfig,
    ConversationId,
    StoredMessage,
    StoredToolCall,
    StoredToolCallFunction,
)
from base.models.view_models.llm_result import LLMResult
from base.models.view_models.streams import StreamEnd, StreamMessage, StreamStart
from base.services.chat_history import ChatHistoryService
from base.services.common.meta import scope_id
from base.services.execution_context import ExecutionContext
from base.services.prompt.loader import PromptLoader
from base.services.tool_calling import Tool, Toolbox
from base.tools.stream import StreamIterator
from chat_api.prompts.templates import test as test_template
from chat_api.services.agents import test as agent_module
from chat_api.stores.chat_history.sql import SqlChatHistoryStore

# Bound here rather than imported: pytest tries to collect a module-level
# `Test*` name as a test class.
chat_agent = agent_module.TestAgent


class RecordingLLM:
    """Answers like an `AbstractLLMService`, keeping every kwarg it is handed."""

    reply: LLMResult = LLMResult(content="  an answer  ")

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def acomplete(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        return type(self).reply

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "no completion was requested"
        return self.calls[-1]


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> type[RecordingLLM]:
    """The agent builds its own provider, so the class is what gets replaced."""
    recorder = type("RecordingLLM", (RecordingLLM,), {})
    monkeypatch.setattr(chat_agent, "LLM_SERVICE", recorder)
    return recorder


@pytest.fixture
def prompts() -> None:
    """`PromptLoader` was purged with the other singletons; re-register."""
    PromptLoader().register(test_template.test_prompt)


@pytest.fixture
def conversation(tables: None) -> ConversationId:
    scope_id.set(uuid.uuid4())
    return ConversationId(uuid.uuid4())


@pytest.fixture
def agent(llm: type[RecordingLLM], prompts: None, conversation: ConversationId) -> Any:
    built = chat_agent()
    ExecutionContext().flow_variables.conversation_id = conversation
    ExecutionContext().flow_variables.user_query = "what is the capital of France?"
    return built


def sent(agent: Any) -> dict[str, Any]:
    service = agent._llm_service
    assert isinstance(service, RecordingLLM)
    return service.last


def append(conversation: ConversationId, role: MessageRole, content: str) -> None:
    ChatHistoryService(ChatHistoryConfig(), SqlChatHistoryStore).append_message(
        conversation, role=role, content=content
    )


# Wiring


def test_the_agent_reads_the_history_the_routers_write() -> None:
    assert chat_agent.CHAT_HISTORY_STORE is SqlChatHistoryStore


def test_the_agent_and_the_router_share_one_chat_history_service(
    tables: None,
) -> None:
    """`ChatHistoryService` is keyed on its arguments, the store type among them."""
    from_router = ChatHistoryService(ChatHistoryConfig(), SqlChatHistoryStore)
    from_agent = ChatHistoryService(ChatHistoryConfig(), chat_agent.CHAT_HISTORY_STORE)
    assert from_router is from_agent


def test_the_prompt_the_agent_asks_for_is_one_the_service_ships(
    prompts: None,
) -> None:
    assert chat_agent.PROMPT_KEY == test_template.test_prompt.key
    assert PromptLoader().get(chat_agent.PROMPT_KEY) is test_template.test_prompt


def test_the_agent_is_one_instance_per_process(agent: Any) -> None:
    assert chat_agent() is agent


# Running


async def test_the_answer_is_what_the_model_returned(agent: Any) -> None:
    await agent.arun()
    assert ExecutionContext().flow_variables.final_answer == "an answer"


async def test_an_empty_reply_becomes_an_empty_answer(
    agent: Any, llm: type[RecordingLLM]
) -> None:
    llm.reply = LLMResult(content=None)
    await agent.arun()
    assert ExecutionContext().flow_variables.final_answer == ""


async def test_the_question_reaches_the_rendered_prompt(agent: Any) -> None:
    await agent.arun()

    prompt = sent(agent)["prompt"]
    assert [message.content for message in prompt.messages] == [
        "what is the capital of France?"
    ]


async def test_the_system_instruction_comes_from_the_template(agent: Any) -> None:
    await agent.arun()

    prompt = sent(agent)["prompt"]
    assert [i.content for i in prompt.instructions] == [
        test_template.TEST_SYSTEM_PROMPT
    ]


async def test_the_output_ceiling_the_agent_declares_reaches_the_provider(
    agent: Any,
) -> None:
    await agent.arun()
    assert sent(agent)["max_output_tokens"] == chat_agent.MAX_OUTPUT_TOKENS


async def test_an_ordinary_turn_asks_for_no_streaming(agent: Any) -> None:
    await agent.arun()
    assert sent(agent)["stream_iterator"] is None


# History


async def test_earlier_turns_are_spliced_into_the_prompt(
    agent: Any, conversation: ConversationId
) -> None:
    append(conversation, MessageRole.USER, "hello")
    append(conversation, MessageRole.ASSISTANT, "hi there")
    await agent.arun()

    prompt = sent(agent)["prompt"]
    assert [m.content for m in prompt.messages] == [
        "hello",
        "hi there",
        "what is the capital of France?",
    ]


async def test_the_current_question_is_not_repeated_as_history(
    agent: Any, conversation: ConversationId
) -> None:
    """The router stores the question before the agent runs; the template
    substitutes it already."""
    append(conversation, MessageRole.ASSISTANT, "hi there")
    append(conversation, MessageRole.USER, "what is the capital of France?")
    await agent.arun()

    prompt = sent(agent)["prompt"]
    assert [m.content for m in prompt.messages] == [
        "hi there",
        "what is the capital of France?",
    ]


async def test_a_conversation_with_no_history_still_runs(agent: Any) -> None:
    await agent.arun()
    assert len(sent(agent)["prompt"].messages) == 1


async def test_another_conversations_history_never_reaches_the_prompt(
    agent: Any, conversation: ConversationId
) -> None:
    other = ConversationId(uuid.uuid4())
    SqlChatHistoryStore().append(
        other,
        [
            StoredMessage(
                conversation_id=other,
                sequence=1,
                role=MessageRole.USER,
                content="somebody else's question",
            )
        ],
    )
    await agent.arun()

    prompt = sent(agent)["prompt"]
    assert [m.content for m in prompt.messages] == ["what is the capital of France?"]


async def test_history_is_capped_at_the_limit_the_agent_declares(
    agent: Any, conversation: ConversationId
) -> None:
    for i in range(chat_agent.HISTORY_LIMIT + 10):
        append(conversation, MessageRole.USER, f"q{i}")
    await agent.arun()

    prompt = sent(agent)["prompt"]
    assert len(prompt.messages) == chat_agent.HISTORY_LIMIT
    assert prompt.messages[0].content == "q10"


# Tools


class ScriptedLLM(RecordingLLM):
    """Answers each call with the next reply, streaming it as a provider would."""

    replies: list[LLMResult] = []

    async def acomplete(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        reply = type(self).replies[min(len(self.calls), len(type(self).replies)) - 1]
        stream = kwargs.get("stream_iterator")
        if stream is not None:
            await stream.asend(StreamStart())
            if reply.content:
                await stream.asend(StreamMessage(data=reply.content))
            await stream.asend(StreamEnd())
        return reply


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> type[ScriptedLLM]:
    recorder = type("ScriptedLLM", (ScriptedLLM,), {"replies": []})
    monkeypatch.setattr(chat_agent, "LLM_SERVICE", recorder)
    return recorder


@pytest.fixture
def tool_agent(
    scripted: type[ScriptedLLM],
    prompts: None,
    conversation: ConversationId,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """The agent with one MCP-shaped tool on offer."""

    async def whoami(arguments: dict[str, Any]) -> str:
        del arguments
        return "caller=ada tenant=acme"

    tools = Toolbox(
        Tool.from_json_schema(
            name="mcp__whoami",
            description="Who the call runs as.",
            schema={"type": "object", "properties": {}},
            handler=whoami,
        )
    )

    async def atoolbox() -> Toolbox:
        return tools

    monkeypatch.setattr(agent_module.MCP, "atoolbox", atoolbox)
    built = chat_agent()
    ExecutionContext().flow_variables.conversation_id = conversation
    ExecutionContext().flow_variables.user_query = "who am I?"
    return built


def whoami_call() -> StoredToolCall:
    return StoredToolCall(
        id="call-1", function=StoredToolCallFunction(name="mcp__whoami", arguments="{}")
    )


async def test_the_mcp_tools_are_offered_to_the_model(
    tool_agent: Any, scripted: type[ScriptedLLM]
) -> None:
    scripted.replies = [LLMResult(content="hi")]
    await tool_agent.arun()
    assert [tool.name for tool in sent(tool_agent)["tools"]] == ["mcp__whoami"]


async def test_a_tool_the_model_asks_for_is_run_and_its_result_answers(
    tool_agent: Any, scripted: type[ScriptedLLM]
) -> None:
    scripted.replies = [
        LLMResult(content=None, tool_calls=[whoami_call()]),
        LLMResult(content="You are ada, in acme."),
    ]
    await tool_agent.arun()

    assert ExecutionContext().flow_variables.final_answer == "You are ada, in acme."
    second = tool_agent._llm_service.calls[1]
    assert second["prompt"].messages[-1].content == "who am I?"
    assert [m.content for m in second["messages"] if m.role == MessageRole.TOOL] == [
        "caller=ada tenant=acme"
    ]


async def test_a_streamed_turn_with_a_tool_round_streams_the_answer_and_ends_once(
    tool_agent: Any, scripted: type[ScriptedLLM]
) -> None:
    scripted.replies = [
        LLMResult(content=None, tool_calls=[whoami_call()]),
        LLMResult(content="You are ada."),
    ]
    stream = StreamIterator()
    context = ExecutionContext()
    context.streaming = True
    context.stream_iterator = stream

    await tool_agent.arun()

    frames = [json.loads(str(frame["data"])) async for frame in stream]  # type: ignore[index]
    assert [f["message_type"] for f in frames] == ["start", "stream", "end"]
    assert frames[1]["data"] == "You are ada."
