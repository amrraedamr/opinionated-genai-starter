"""`StoredMessage` and `StoredToolCall`: the shape both stores round-trip.

The validator is the only thing standing between a backend-neutral store and a
message no provider could replay, so each branch of it is pinned — including
the gaps, which are marked.
"""

from uuid import UUID

import pytest
from pydantic import ValidationError

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import (
    ChatHistoryConfig,
    StoredMessage,
    StoredToolCall,
    StoredToolCallFunction,
)

CONVERSATION = UUID(int=1)


def message(**kwargs: object) -> StoredMessage:
    return StoredMessage(conversation_id=CONVERSATION, sequence=1, **kwargs)  # type: ignore[arg-type]


def a_tool_call(identifier: str = "call_1") -> StoredToolCall:
    return StoredToolCall(
        id=identifier,
        function=StoredToolCallFunction(
            name="get_weather", arguments='{"city":"Cairo"}'
        ),
    )


def test_a_tool_call_holds_its_arguments_as_a_json_string() -> None:
    call = a_tool_call()
    assert call.type == "function"
    assert call.function.arguments == '{"city":"Cairo"}'
    assert isinstance(call.function.arguments, str)


def test_a_tool_call_of_any_other_type_is_refused() -> None:
    with pytest.raises(ValidationError):
        StoredToolCall(
            id="c",
            type="custom",  # type: ignore[arg-type]
            function=StoredToolCallFunction(name="n", arguments="{}"),
        )


def test_a_fresh_draft_has_no_id_until_a_store_appends_it() -> None:
    assert message(role=MessageRole.USER, content="hi").id is None


def test_a_message_carries_its_conversation_and_its_place_in_it() -> None:
    stored = message(role=MessageRole.USER, content="hi")
    assert stored.conversation_id == CONVERSATION
    assert stored.sequence == 1
    assert stored.metadata == {}


@pytest.mark.parametrize("role", [MessageRole.SYSTEM, MessageRole.DEVELOPER])
def test_an_instruction_message_requires_content(role: MessageRole) -> None:
    with pytest.raises(ValidationError, match="require content"):
        message(role=role)


def test_a_user_message_requires_content_or_parts() -> None:
    with pytest.raises(ValidationError, match="require content or parts"):
        message(role=MessageRole.USER)


def test_a_user_message_may_carry_parts_instead_of_content() -> None:
    stored = message(
        role=MessageRole.USER,
        parts=[{"type": "text", "text": "hello"}],
    )
    assert stored.content is None
    assert stored.parts is not None and len(stored.parts) == 1


def test_an_assistant_message_may_be_tool_calls_with_no_content() -> None:
    stored = message(role=MessageRole.ASSISTANT, tool_calls=[a_tool_call()])
    assert stored.content is None
    assert stored.tool_calls is not None and stored.tool_calls[0].id == "call_1"


def test_an_assistant_message_with_neither_content_nor_tool_calls_is_refused() -> None:
    with pytest.raises(ValidationError, match="require content, tool_calls, or both"):
        message(role=MessageRole.ASSISTANT)


def test_a_tool_message_requires_the_call_it_answers() -> None:
    with pytest.raises(ValidationError, match="Tool messages require tool_call_id"):
        message(role=MessageRole.TOOL, content="21")


def test_a_tool_message_may_not_also_issue_tool_calls() -> None:
    with pytest.raises(ValidationError, match="Only assistant messages"):
        message(
            role=MessageRole.TOOL, tool_call_id="call_1", tool_calls=[a_tool_call()]
        )


def test_a_tool_message_pairs_with_its_assistant_turn() -> None:
    stored = message(role=MessageRole.TOOL, content="21", tool_call_id="call_1")
    assert stored.tool_call_id == "call_1"


def test_a_user_message_is_allowed_to_carry_a_tool_call_id() -> None:
    """Pinned, not endorsed: the `role != TOOL` guard sits behind an earlier
    `elif`, so only a tool-role message ever reaches it. See the report."""
    assert message(role=MessageRole.USER, content="hi", tool_call_id="call_1")


def test_an_assistant_message_is_allowed_to_carry_a_tool_call_id() -> None:
    """Pinned, not endorsed. Same unreachable guard as the user case."""
    assert message(role=MessageRole.ASSISTANT, content="hi", tool_call_id="call_1")


def test_a_user_message_is_allowed_to_carry_tool_calls() -> None:
    """Pinned, not endorsed: `Only assistant messages may define tool_calls`
    is only reachable for a tool-role message."""
    assert message(role=MessageRole.USER, content="hi", tool_calls=[a_tool_call()])


def test_a_message_keeps_a_cached_token_count_when_one_was_computed() -> None:
    assert message(role=MessageRole.USER, content="hi", token_count=3).token_count == 3


def test_a_message_camelises_its_nested_tool_calls_on_dump() -> None:
    dumped = message(role=MessageRole.ASSISTANT, tool_calls=[a_tool_call()]).serialize(
        exclude_none=True
    )
    assert dumped["conversationId"] == str(CONVERSATION)
    assert dumped["toolCalls"][0]["function"]["name"] == "get_weather"


def test_each_message_gets_its_own_metadata_mapping() -> None:
    first = message(role=MessageRole.USER, content="a")
    second = message(role=MessageRole.USER, content="b")
    first.metadata["is_policy_violated"] = True
    assert second.metadata == {}


def test_the_chat_history_config_defaults_to_a_tokenizer_tiktoken_knows() -> None:
    assert ChatHistoryConfig().tokenizer_model == "gpt-4o"
