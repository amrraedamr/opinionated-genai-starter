"""`ChatHistoryService` owns sequencing, token counting and the tool contract.

It composes a store *type* and a `TokenCounter` from one config object rather
than taking either as an instance, because it is a `KeyedSingleton` and an
instance in the key would not hash. Everything a conversation is read back
through — the window, visibility, orphaned tool messages — lives here too, so
a store never has to know the rules.
"""

from __future__ import annotations

import uuid

import pytest

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import (
    ChatHistoryConfig,
    ConversationId,
    StoredToolCall,
    StoredToolCallFunction,
)
from base.services.chat_history import ChatHistoryService
from base.services.store.chat_history.memory import InMemoryChatHistoryStore
from base.tools.tokenizer import TokenCounter


def fresh_service() -> tuple[ChatHistoryService, ConversationId]:
    """A service over a store no other test shares, plus an empty conversation."""
    store = type(f"Store{uuid.uuid4().hex[:8]}", (InMemoryChatHistoryStore,), {})
    service = ChatHistoryService(ChatHistoryConfig(), store)
    return service, ConversationId(uuid.uuid4())


def tool_call(identifier: str = "call_1", name: str = "get_weather") -> StoredToolCall:
    return StoredToolCall(
        id=identifier,
        function=StoredToolCallFunction(name=name, arguments='{"city":"Doha"}'),
    )


def test_the_service_is_keyed_on_its_config_and_store_type() -> None:
    class SharedStore(InMemoryChatHistoryStore):
        pass

    first = ChatHistoryService(ChatHistoryConfig(), SharedStore)
    again = ChatHistoryService(ChatHistoryConfig(), SharedStore)
    assert first is again

    class OtherStore(InMemoryChatHistoryStore):
        pass

    assert ChatHistoryService(ChatHistoryConfig(), OtherStore) is not first


def test_the_service_builds_its_own_store_and_counter() -> None:
    class OwnedStore(InMemoryChatHistoryStore):
        pass

    service = ChatHistoryService(
        ChatHistoryConfig(tokenizer_model="gpt-4o"), OwnedStore
    )

    assert isinstance(service._store, OwnedStore)
    assert isinstance(service._token_counter, TokenCounter)
    assert service._token_counter is TokenCounter(model="gpt-4o")


async def test_appending_assigns_sequences_from_one_upwards() -> None:
    service, conversation = fresh_service()

    first = await service.aappend_message(
        conversation, role=MessageRole.USER, content="one"
    )
    second = await service.aappend_message(
        conversation, role=MessageRole.ASSISTANT, content="two"
    )

    assert (first.sequence, second.sequence) == (1, 2)


async def test_messages_come_back_in_sequence_order() -> None:
    service, conversation = fresh_service()
    for index in range(4):
        await service.aappend_message(
            conversation, role=MessageRole.ASSISTANT, content=f"m{index}"
        )

    fetched = await service.aget_messages(conversation)
    assert [m.sequence for m in fetched] == [1, 2, 3, 4]
    assert [m.content for m in fetched] == ["m0", "m1", "m2", "m3"]


async def test_a_limit_returns_the_most_recent_window_still_ascending() -> None:
    service, conversation = fresh_service()
    for index in range(6):
        await service.aappend_message(
            conversation, role=MessageRole.ASSISTANT, content=f"m{index}"
        )

    fetched = await service.aget_messages(conversation, limit=2)
    assert [m.content for m in fetched] == ["m4", "m5"]


async def test_no_limit_returns_the_whole_conversation() -> None:
    service, conversation = fresh_service()
    for index in range(3):
        await service.aappend_message(
            conversation, role=MessageRole.ASSISTANT, content=f"m{index}"
        )

    assert len(await service.aget_messages(conversation)) == 3


async def test_a_message_is_stored_with_its_token_count() -> None:
    service, conversation = fresh_service()
    content = "Tokens are counted at append time, not at read time."

    stored = await service.aappend_message(
        conversation, role=MessageRole.USER, content=content
    )

    assert stored.token_count == TokenCounter("gpt-4o").count(content)
    assert stored.token_count > 0


async def test_an_empty_assistant_turn_with_tool_calls_counts_the_call() -> None:
    service, conversation = fresh_service()
    call = tool_call()

    stored = await service.aappend_message(
        conversation, role=MessageRole.ASSISTANT, tool_calls=[call]
    )

    counter = TokenCounter("gpt-4o")
    expected = counter.count(call.function.name) + counter.count(
        call.function.arguments
    )
    assert stored.token_count == expected


async def test_a_longer_message_counts_more_tokens_than_a_shorter_one() -> None:
    service, conversation = fresh_service()

    short = await service.aappend_message(
        conversation, role=MessageRole.USER, content="hi"
    )
    long = await service.aappend_message(
        conversation,
        role=MessageRole.USER,
        content="hi " * 50,
    )

    assert long.token_count > short.token_count


async def test_a_policy_violated_message_is_hidden_from_reads() -> None:
    service, conversation = fresh_service()
    await service.aappend_message(conversation, role=MessageRole.USER, content="ok")
    await service.aappend_message(
        conversation,
        role=MessageRole.USER,
        content="blocked",
        metadata={"is_policy_violated": True},
    )

    assert [m.content for m in await service.aget_messages(conversation)] == ["ok"]


async def test_an_orphaned_tool_message_is_dropped() -> None:
    service, conversation = fresh_service()
    await service.aappend_message(
        conversation, role=MessageRole.USER, content="weather?"
    )
    await service.aappend_message(
        conversation,
        role=MessageRole.TOOL,
        content="21",
        tool_call_id="call_nobody_made",
    )

    assert [m.role for m in await service.aget_messages(conversation)] == [
        MessageRole.USER
    ]


async def test_an_assistant_missing_its_tool_result_is_demoted() -> None:
    service, conversation = fresh_service()
    await service.aappend_message(
        conversation,
        role=MessageRole.ASSISTANT,
        content="calling",
        tool_calls=[tool_call()],
    )

    fetched = await service.aget_messages(conversation)
    assert len(fetched) == 1
    assert fetched[0].tool_calls is None


async def test_a_complete_tool_pair_survives_intact() -> None:
    service, conversation = fresh_service()
    call = tool_call()
    await service.aappend_message(
        conversation, role=MessageRole.ASSISTANT, content=None, tool_calls=[call]
    )
    await service.aappend_message(
        conversation, role=MessageRole.TOOL, content="21", tool_call_id=call.id
    )

    fetched = await service.aget_messages(conversation)
    assert [m.role for m in fetched] == [MessageRole.ASSISTANT, MessageRole.TOOL]
    assert fetched[0].tool_calls == [call]


async def test_clearing_a_conversation_empties_it() -> None:
    service, conversation = fresh_service()
    await service.aappend_message(conversation, role=MessageRole.USER, content="hi")

    await service.aclear(conversation)

    assert await service.aget_messages(conversation) == []


async def test_conversations_do_not_see_each_other() -> None:
    service, first = fresh_service()
    second = ConversationId(uuid.uuid4())

    await service.aappend_message(first, role=MessageRole.USER, content="mine")
    await service.aappend_message(second, role=MessageRole.USER, content="theirs")

    assert [m.content for m in await service.aget_messages(first)] == ["mine"]
    assert [m.content for m in await service.aget_messages(second)] == ["theirs"]


def test_the_sync_path_assigns_the_same_sequences() -> None:
    service, conversation = fresh_service()

    first = service.append_message(conversation, role=MessageRole.USER, content="one")
    second = service.append_message(
        conversation, role=MessageRole.ASSISTANT, content="two"
    )

    assert (first.sequence, second.sequence) == (1, 2)
    assert [m.content for m in service.get_messages(conversation)] == ["one", "two"]


async def test_a_message_role_that_cannot_be_stored_is_rejected() -> None:
    service, conversation = fresh_service()

    with pytest.raises(ValueError, match="System messages require content"):
        await service.aappend_message(
            conversation, role=MessageRole.SYSTEM, content=None
        )
