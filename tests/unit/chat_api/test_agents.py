"""`TestAgent` — the one agent this service ships, and the shape a copy takes.

Most of what it is lives in its ClassVars, and one of them is load-bearing:
`ChatHistoryService` is keyed by the store *type*, so an agent naming a
different store than the routers would read a different conversation and
answer from an empty history.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import (
    ChatHistoryConfig,
    ConversationId,
    StoredMessage,
)
from base.models.view_models.llm_result import LLMResult
from base.services.chat_history import ChatHistoryService
from base.services.common.meta import scope_id
from base.services.execution_context import ExecutionContext
from base.services.prompt.loader import PromptLoader
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
