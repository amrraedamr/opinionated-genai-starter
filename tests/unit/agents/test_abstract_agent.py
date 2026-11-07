"""`AbstractAgent`'s ClassVars are the agent's whole contract with its provider.

Every knob an agent sets as a ClassVar has to arrive at the LLM service, and
has to lose to an explicit keyword on the call. Both halves are pinned here:
the defaults were once named as explicit arguments as well as applied through
`setdefault`, so `max_output_tokens=` at a call site raised
`TypeError: got multiple values for keyword argument`.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from base.models.definitions.canned_messages import CannedMessages
from base.models.definitions.message import MessageRole
from base.models.definitions.model import ReasoningEffort
from base.models.view_models.chat_history import ChatHistoryConfig, ConversationId
from base.models.view_models.exceptions.llm import LLMError, LLMRateLimitError
from base.models.view_models.flow import FlowAbortError
from base.models.view_models.model import CacheHint, CachePersistence
from base.models.view_models.streams import StreamEnd, StreamMessage
from base.services.agents import AbstractAgent
from base.services.chat_history import ChatHistoryService
from base.services.common.meta import SingletonABC, scope_id
from base.services.llm.openai import OpenAIService
from base.services.store.chat_history.memory import InMemoryChatHistoryStore
from base.tools.stream import StreamIterator
from tests.fakes import build_prompt
from tests.unit.agents.fake_llm import Answer, RecordingLLM, build_agent, llm_of


def test_an_agent_without_execute_cannot_be_instantiated() -> None:
    incomplete = SingletonABC("Incomplete", (AbstractAgent,), {})
    with pytest.raises(TypeError, match="_execute"):
        incomplete()


def test_label_defaults_to_the_agent_class_name() -> None:
    agent = build_agent()
    kwargs: dict[str, Any] = {}
    agent._apply_defaults(kwargs, None)
    assert kwargs["label"] == type(agent).__name__


async def test_class_var_defaults_reach_the_completion_call() -> None:
    agent = build_agent(
        MAX_OUTPUT_TOKENS=321,
        REASONING_EFFORT=ReasoningEffort.HIGH,
        CACHE_HINT=CacheHint(persistence=CachePersistence.LONG),
    )
    await agent._acomplete(prompt=build_prompt())

    sent = llm_of(agent).last_complete
    assert sent["max_output_tokens"] == 321
    assert sent["reasoning_effort"] is ReasoningEffort.HIGH
    assert sent["cache_hint"] == CacheHint(persistence=CachePersistence.LONG)
    assert sent["label"] == type(agent).__name__


async def test_class_var_defaults_reach_the_parse_call() -> None:
    agent = build_agent(
        MAX_OUTPUT_TOKENS=222,
        REASONING_EFFORT=ReasoningEffort.LOW,
        CACHE_HINT=CacheHint(persistence=CachePersistence.SHORT),
    )
    await agent._aparse(Answer, prompt=build_prompt())

    sent = llm_of(agent).last_parse
    assert sent["schema"] is Answer
    assert sent["max_output_tokens"] == 222
    assert sent["reasoning_effort"] is ReasoningEffort.LOW
    assert sent["cache_hint"] == CacheHint(persistence=CachePersistence.SHORT)
    assert sent["label"] == type(agent).__name__


@pytest.mark.parametrize(
    ("keyword", "override"),
    [
        ("max_output_tokens", 7),
        ("reasoning_effort", ReasoningEffort.NONE),
        ("cache_hint", None),
    ],
)
async def test_an_explicit_keyword_beats_the_class_var_on_complete(
    keyword: str, override: Any
) -> None:
    agent = build_agent(
        MAX_OUTPUT_TOKENS=1000,
        REASONING_EFFORT=ReasoningEffort.HIGH,
        CACHE_HINT=CacheHint(persistence=CachePersistence.LONG),
    )
    await agent._acomplete(prompt=build_prompt(), **{keyword: override})

    assert llm_of(agent).last_complete[keyword] == override


@pytest.mark.parametrize(
    ("keyword", "override"),
    [
        ("max_output_tokens", 9),
        ("reasoning_effort", ReasoningEffort.NONE),
        ("cache_hint", None),
    ],
)
async def test_an_explicit_keyword_beats_the_class_var_on_parse(
    keyword: str, override: Any
) -> None:
    agent = build_agent(
        MAX_OUTPUT_TOKENS=1000,
        REASONING_EFFORT=ReasoningEffort.HIGH,
        CACHE_HINT=CacheHint(persistence=CachePersistence.LONG),
    )
    await agent._aparse(Answer, prompt=build_prompt(), **{keyword: override})

    assert llm_of(agent).last_parse[keyword] == override


async def test_an_explicit_label_beats_the_class_name_on_both_calls() -> None:
    agent = build_agent()
    await agent._acomplete(prompt=build_prompt(), label="Agent:0001")
    await agent._aparse(Answer, prompt=build_prompt(), label="Agent:0002")

    llm = llm_of(agent)
    assert llm.last_complete["label"] == "Agent:0001"
    assert llm.last_parse["label"] == "Agent:0002"


async def test_prompt_messages_and_tools_are_forwarded_untouched() -> None:
    agent = build_agent()
    prompt = build_prompt()
    tools: list[Any] = []
    await agent._acomplete(prompt=prompt, messages=[], tools=tools)

    sent = llm_of(agent).last_complete
    assert sent["prompt"] is prompt
    assert sent["messages"] == []
    assert sent["tools"] is tools


async def test_a_non_streaming_request_sends_no_stream_iterator() -> None:
    agent = build_agent()
    await agent._acomplete(prompt=build_prompt())

    assert llm_of(agent).last_complete["stream_iterator"] is None


async def test_a_streaming_request_sends_the_contexts_iterator() -> None:
    agent = build_agent()
    iterator = StreamIterator()
    agent._execution_context.streaming = True
    agent._execution_context.stream_iterator = iterator

    await agent._acomplete(prompt=build_prompt())

    assert llm_of(agent).last_complete["stream_iterator"] is iterator


async def test_an_explicit_stream_iterator_beats_the_context() -> None:
    agent = build_agent()
    agent._execution_context.streaming = True
    agent._execution_context.stream_iterator = StreamIterator()
    explicit = StreamIterator()

    await agent._acomplete(prompt=build_prompt(), stream_iterator=explicit)

    assert llm_of(agent).last_complete["stream_iterator"] is explicit


async def test_a_parsed_call_carries_no_stream_iterator_at_all() -> None:
    agent = build_agent()
    agent._execution_context.streaming = True
    agent._execution_context.stream_iterator = StreamIterator()

    await agent._aparse(Answer, prompt=build_prompt())

    assert "stream_iterator" not in llm_of(agent).last_parse


def test_the_default_provider_is_openai_and_is_built_with_no_arguments() -> None:
    assert AbstractAgent.LLM_SERVICE is OpenAIService
    assert AbstractAgent.MODEL_CONFIGURATIONS is None


def test_model_configurations_are_passed_to_the_provider_when_set() -> None:
    seen: list[Any] = []

    class ConfigurableLLM(RecordingLLM):
        def __init__(self, *args: Any) -> None:
            super().__init__()
            seen.append(args)

    marker = object()
    build_agent(LLM_SERVICE=ConfigurableLLM, MODEL_CONFIGURATIONS=marker)

    assert seen == [(marker,)]


def test_an_unset_model_configuration_constructs_the_provider_bare() -> None:
    seen: list[Any] = []

    class ConfigurableLLM(RecordingLLM):
        def __init__(self, *args: Any) -> None:
            super().__init__()
            seen.append(args)

    build_agent(LLM_SERVICE=ConfigurableLLM)

    assert seen == [()]


def test_prompt_key_is_empty_until_an_agent_names_one() -> None:
    assert AbstractAgent.PROMPT_KEY == ""
    assert build_agent(PROMPT_KEY="answer").PROMPT_KEY == "answer"


def test_history_limit_defaults_to_forty() -> None:
    assert AbstractAgent.HISTORY_LIMIT == 40


def test_the_default_chat_history_store_is_the_in_memory_reference() -> None:
    assert AbstractAgent.CHAT_HISTORY_STORE is InMemoryChatHistoryStore


def test_the_agent_composes_a_chat_history_service_over_its_store() -> None:
    class OwnStore(InMemoryChatHistoryStore):
        pass

    agent = build_agent(CHAT_HISTORY_STORE=OwnStore)

    assert isinstance(agent._chat_history_service, ChatHistoryService)
    assert agent._chat_history_service is ChatHistoryService(
        ChatHistoryConfig(), OwnStore
    )


def test_the_execution_context_is_resolved_per_request_not_frozen() -> None:
    agent = build_agent()
    first = agent._execution_context
    assert agent._execution_context is first

    token = scope_id.set(uuid.uuid4())
    try:
        assert agent._execution_context is not first
    finally:
        scope_id.reset(token)


async def test_no_conversation_means_no_history() -> None:
    agent = build_agent()
    agent._execution_context.flow_variables.conversation_id = None

    assert await agent._aget_history() == []


async def test_history_drops_the_trailing_user_turn() -> None:
    class TrailingStore(InMemoryChatHistoryStore):
        pass

    agent = build_agent(CHAT_HISTORY_STORE=TrailingStore)
    conversation = ConversationId(uuid.uuid4())
    agent._execution_context.flow_variables.conversation_id = conversation

    history = agent._chat_history_service
    await history.aappend_message(conversation, role=MessageRole.USER, content="first")
    await history.aappend_message(
        conversation, role=MessageRole.ASSISTANT, content="answer"
    )
    await history.aappend_message(
        conversation, role=MessageRole.USER, content="current query"
    )

    fetched = await agent._aget_history()
    assert [m.content for m in fetched] == ["first", "answer"]


async def test_history_keeps_a_trailing_assistant_turn() -> None:
    class AssistantTailStore(InMemoryChatHistoryStore):
        pass

    agent = build_agent(CHAT_HISTORY_STORE=AssistantTailStore)
    conversation = ConversationId(uuid.uuid4())
    agent._execution_context.flow_variables.conversation_id = conversation

    history = agent._chat_history_service
    await history.aappend_message(conversation, role=MessageRole.USER, content="q")
    await history.aappend_message(conversation, role=MessageRole.ASSISTANT, content="a")

    assert [m.content for m in await agent._aget_history()] == ["q", "a"]


async def test_history_limit_is_handed_to_the_store_as_the_window() -> None:
    seen: list[int | None] = []

    class WindowStore(InMemoryChatHistoryStore):
        async def _aget(self, parent_id: Any, *, limit: int | None = None) -> list[Any]:
            seen.append(limit)
            return await super()._aget(parent_id, limit=limit)

    agent = build_agent(CHAT_HISTORY_STORE=WindowStore, HISTORY_LIMIT=3)
    conversation = ConversationId(uuid.uuid4())
    agent._execution_context.flow_variables.conversation_id = conversation

    history = agent._chat_history_service
    for index in range(5):
        await history.aappend_message(
            conversation, role=MessageRole.ASSISTANT, content=f"m{index}"
        )

    fetched = await agent._aget_history()
    assert seen == [3]
    assert [m.content for m in fetched] == ["m2", "m3", "m4"]


async def test_run_calls_the_three_hooks_in_order() -> None:
    order: list[str] = []

    async def _pre(self: AbstractAgent) -> None:
        order.append("pre")

    async def _exec(self: AbstractAgent) -> None:
        order.append("execute")

    async def _post(self: AbstractAgent) -> None:
        order.append("post")

    agent = build_agent(_pre_execute=_pre, _execute=_exec, _post_execute=_post)
    await agent.arun()

    assert order == ["pre", "execute", "post"]


async def test_a_failing_agent_aborts_the_flow_with_a_canned_message() -> None:
    async def _exec(self: AbstractAgent) -> None:
        raise LLMRateLimitError("slow down", attempts=3)

    agent = build_agent(_execute=_exec)

    with pytest.raises(FlowAbortError) as caught:
        await agent.arun()

    assert caught.value.canned_message is CannedMessages.LLM_RATE_LIMITED
    assert isinstance(caught.value.cause, LLMRateLimitError)


async def test_post_execute_is_skipped_when_execute_fails() -> None:
    ran: list[str] = []

    async def _exec(self: AbstractAgent) -> None:
        raise LLMError("boom")

    async def _post(self: AbstractAgent) -> None:
        ran.append("post")

    agent = build_agent(_execute=_exec, _post_execute=_post)

    with pytest.raises(FlowAbortError):
        await agent.arun()

    assert ran == []


async def test_a_streaming_failure_sends_the_canned_message_then_ends() -> None:
    async def _exec(self: AbstractAgent) -> None:
        raise LLMError("boom")

    agent = build_agent(_execute=_exec)
    iterator = StreamIterator()
    agent._execution_context.streaming = True
    agent._execution_context.stream_iterator = iterator

    with pytest.raises(FlowAbortError):
        await agent.arun()

    first = await iterator._queue.get()  # type: ignore[union-attr]
    second = await iterator._queue.get()  # type: ignore[union-attr]
    assert isinstance(first, StreamMessage)
    assert (
        first.data
        == CannedMessages.LLM_GENERIC_ERROR.value[
            agent._execution_context.flow_variables.locale
        ]
    )
    assert isinstance(second, StreamEnd)


async def test_a_non_llm_exception_is_not_converted_to_a_canned_message() -> None:
    async def _exec(self: AbstractAgent) -> None:
        raise ZeroDivisionError("not an LLM problem")

    agent = build_agent(_execute=_exec)

    with pytest.raises(ZeroDivisionError):
        await agent.arun()
