"""History is stored raw, so what the model is shown of it is cleaned each turn."""

import uuid

import pytest

from base.models.definitions.guardrails import (
    BLOCKED_HISTORY_PLACEHOLDER,
    GuardrailStage,
)
from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import ConversationId, StoredMessage
from base.models.view_models.guardrails import GuardrailVerdict
from base.services.agents import AbstractAgent
from base.services.configuration.mixins import RedisSettingsMixin
from base.services.guardrails import AbstractGuardrail, GuardrailBlocked
from base.services.guardrails.pipeline import GuardrailPipeline
from tests.conftest import MakeSettings
from tests.unit.agents.fake_llm import build_agent, llm_of

CONVERSATION = ConversationId(uuid.UUID(int=7))


class Counting(AbstractGuardrail):
    """Blocks one word, redacts another, and counts how often it is asked."""

    def __init__(self) -> None:
        self.calls = 0

    def _name(self) -> str:
        return "counting"

    def _stages(self) -> frozenset[GuardrailStage]:
        return frozenset({GuardrailStage.INPUT})

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        self.calls += 1
        if "forbidden" in content:
            return GuardrailVerdict.block("refused", category="test")
        if "secret" in content:
            return GuardrailVerdict.redact(content.replace("secret", "[gone]"))
        return GuardrailVerdict.allow()

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return self._check(content, stage)


def message(role: MessageRole, content: str, sequence: int) -> StoredMessage:
    return StoredMessage(
        conversation_id=CONVERSATION,
        role=role,
        content=content,
        sequence=sequence,
    )


@pytest.fixture
def guarded(make_settings: MakeSettings) -> tuple[AbstractAgent, Counting]:
    make_settings()
    counting = Counting()
    agent = build_agent()
    agent._guardrail_pipeline = GuardrailPipeline(counting)
    return agent, counting


async def sent(agent: AbstractAgent, history: list[StoredMessage]) -> list[str]:
    """What the model was shown of `history`."""
    await agent._acomplete(messages=history)
    shown: list[StoredMessage] = llm_of(agent).last_complete["messages"]
    return [m.content or "" for m in shown]


async def test_a_blocked_message_in_history_is_withheld_not_fatal(
    guarded: tuple[AbstractAgent, Counting],
) -> None:
    agent, _ = guarded
    history = [
        message(MessageRole.USER, "something forbidden", 1),
        message(MessageRole.ASSISTANT, "I cannot help with that", 2),
        message(MessageRole.USER, "ok, something else", 3),
    ]

    assert await sent(agent, history) == [
        BLOCKED_HISTORY_PLACEHOLDER,
        "I cannot help with that",
        "ok, something else",
    ]


async def test_the_newest_message_still_blocks_the_request(
    guarded: tuple[AbstractAgent, Counting],
) -> None:
    agent, _ = guarded
    history = [
        message(MessageRole.USER, "hello", 1),
        message(MessageRole.USER, "something forbidden", 2),
    ]

    with pytest.raises(GuardrailBlocked):
        await agent._acomplete(messages=history)
    assert not llm_of(agent).complete_calls


async def test_personal_data_in_history_is_redacted_every_turn(
    guarded: tuple[AbstractAgent, Counting],
) -> None:
    agent, _ = guarded
    history = [
        message(MessageRole.USER, "my secret is here", 1),
        message(MessageRole.USER, "and now a question", 2),
    ]

    assert (await sent(agent, history))[0] == "my [gone] is here"


async def test_the_stored_history_itself_is_left_raw(
    guarded: tuple[AbstractAgent, Counting],
) -> None:
    agent, _ = guarded
    original = message(MessageRole.USER, "my secret is here", 1)

    await sent(agent, [original, message(MessageRole.USER, "next", 2)])

    assert original.content == "my secret is here"


async def test_an_unchanged_message_is_checked_once_across_turns(
    guarded: tuple[AbstractAgent, Counting],
) -> None:
    """Moderation bills per call; history must not be re-billed every turn."""
    agent, counting = guarded
    history = [
        message(MessageRole.USER, "first question", 1),
        message(MessageRole.USER, "second question", 2),
    ]

    await sent(agent, history)
    await sent(agent, [*history, message(MessageRole.USER, "third", 3)])

    assert counting.calls == 3


async def test_the_models_own_turns_are_never_checked(
    guarded: tuple[AbstractAgent, Counting],
) -> None:
    agent, counting = guarded
    await sent(agent, [message(MessageRole.ASSISTANT, "forbidden", 1)])
    assert counting.calls == 0


async def test_an_unreachable_cache_only_costs_the_saving(
    make_settings: MakeSettings,
) -> None:
    """The cache saves a bill; losing it must not fail the turn."""
    make_settings(RedisSettingsMixin, redis_url="redis://localhost:1/0")
    counting = Counting()
    agent = build_agent()
    agent._guardrail_pipeline = GuardrailPipeline(counting)

    shown = await sent(agent, [message(MessageRole.USER, "my secret is here", 1)])

    assert shown == ["my [gone] is here"]
    assert counting.calls == 1
