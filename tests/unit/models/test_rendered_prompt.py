"""`RenderedPrompt` and its two item kinds, and what each refuses to hold.

An instruction block and a conversation message are different things: one is
the head of the prompt that repeats, the other is the tail that never does.
The validators are what keep a caller from mixing them up.
"""

import pytest
from pydantic import ValidationError

from base.models.definitions.message import MessageRole
from base.models.view_models.prompt import (
    RenderedPrompt,
    RenderedPromptInstruction,
    RenderedPromptMessage,
)


def test_an_instruction_accepts_the_system_role() -> None:
    instruction = RenderedPromptInstruction(
        role=MessageRole.SYSTEM, content="Be terse."
    )
    assert instruction.role is MessageRole.SYSTEM
    assert instruction.content == "Be terse."


def test_an_instruction_accepts_the_developer_role() -> None:
    assert (
        RenderedPromptInstruction(role=MessageRole.DEVELOPER, content="x").role
        is MessageRole.DEVELOPER
    )


@pytest.mark.parametrize(
    "role", [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL]
)
def test_an_instruction_refuses_a_conversation_role(role: MessageRole) -> None:
    with pytest.raises(ValidationError):
        RenderedPromptInstruction(role=role, content="x")


@pytest.mark.parametrize("role", [MessageRole.SYSTEM, MessageRole.DEVELOPER])
def test_a_message_refuses_an_instruction_role(role: MessageRole) -> None:
    with pytest.raises(ValidationError, match="cannot use 'system' or 'developer'"):
        RenderedPromptMessage(role=role, content="x")


def test_a_tool_message_requires_the_call_it_answers() -> None:
    with pytest.raises(ValidationError, match="Tool messages require tool_call_id"):
        RenderedPromptMessage(role=MessageRole.TOOL, content="21")


def test_a_tool_message_with_its_call_id_is_accepted() -> None:
    message = RenderedPromptMessage(
        role=MessageRole.TOOL, content="21", tool_call_id="call_1"
    )
    assert message.tool_call_id == "call_1"


@pytest.mark.parametrize("role", [MessageRole.USER, MessageRole.ASSISTANT])
def test_a_non_tool_message_may_not_carry_a_tool_call_id(role: MessageRole) -> None:
    with pytest.raises(ValidationError, match="Only tool messages"):
        RenderedPromptMessage(role=role, content="x", tool_call_id="call_1")


def test_a_user_message_needs_nothing_beyond_its_content() -> None:
    message = RenderedPromptMessage(role=MessageRole.USER, content="Capital of France?")
    assert message.tool_call_id is None
    assert message.name is None


def test_a_message_may_carry_a_name_whatever_its_role() -> None:
    message = RenderedPromptMessage(role=MessageRole.USER, content="x", name="alice")
    assert message.name == "alice"


def test_a_rendered_prompt_starts_empty_rather_than_none() -> None:
    prompt = RenderedPrompt()
    assert prompt.instructions == []
    assert prompt.messages == []
    assert prompt.variables == {}
    assert prompt.key is None
    assert prompt.name is None
    assert prompt.description is None


def test_a_rendered_prompt_keeps_instructions_and_messages_apart() -> None:
    prompt = RenderedPrompt(
        key="greeting",
        variables={"tone": "terse"},
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content="Be terse.")
        ],
        messages=[RenderedPromptMessage(role=MessageRole.USER, content="Hello?")],
    )
    assert [i.role for i in prompt.instructions] == [MessageRole.SYSTEM]
    assert [m.role for m in prompt.messages] == [MessageRole.USER]
    assert prompt.variables == {"tone": "terse"}


def test_a_rendered_prompt_refuses_an_instruction_in_its_message_list() -> None:
    with pytest.raises(ValidationError):
        RenderedPrompt(messages=[{"role": "system", "content": "x"}])  # type: ignore[list-item]


def test_each_rendered_prompt_gets_its_own_empty_collections() -> None:
    first, second = RenderedPrompt(), RenderedPrompt()
    first.variables["tone"] = "terse"
    assert second.variables == {}


def test_a_rendered_prompt_camelises_its_nested_items_on_dump() -> None:
    prompt = RenderedPrompt(
        messages=[
            RenderedPromptMessage(
                role=MessageRole.TOOL, content="21", tool_call_id="call_1"
            )
        ]
    )
    assert prompt.serialize()["messages"][0]["toolCallId"] == "call_1"
