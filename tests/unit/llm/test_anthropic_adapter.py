"""Pins how neutral messages become Anthropic's wire shape.

Anthropic keeps instructions out of the message list, carries a tool result as
a content block inside a user turn rather than as its own role, and rejects two
consecutive turns with the same role — which a parallel tool call produces
naturally, so adjacent same-role turns are folded together.
"""

import uuid

import pytest

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import (
    StoredMessage,
    StoredToolCall,
    StoredToolCallFunction,
)
from base.models.view_models.content import DocumentPart, ImagePart, TextPart
from base.models.view_models.prompt import (
    RenderedPrompt,
    RenderedPromptInstruction,
    RenderedPromptMessage,
)
from base.services.llm.anthropic.adapter import AnthropicMessageAdapter
from tests.fakes import build_prompt, build_tool, stored_message

CONVERSATION = uuid.UUID(int=1)

TAGGED = "hello\U000e0041\u200b"
"""Text carrying a tag character and a zero-width space."""


def assistant_with_tool_calls(*names: str) -> StoredMessage:
    """One assistant turn asking for each named tool."""
    return StoredMessage(
        conversation_id=CONVERSATION,
        sequence=2,
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=[
            StoredToolCall(
                id=f"toolu_{index}",
                function=StoredToolCallFunction(
                    name=name, arguments='{"city": "Paris"}'
                ),
            )
            for index, name in enumerate(names)
        ],
    )


def tool_result(call_id: str, content: str, sequence: int) -> StoredMessage:
    return StoredMessage(
        conversation_id=CONVERSATION,
        sequence=sequence,
        role=MessageRole.TOOL,
        content=content,
        tool_call_id=call_id,
    )


def test_an_adapter_call_with_neither_input_is_refused() -> None:
    with pytest.raises(ValueError, match="At least one of"):
        AnthropicMessageAdapter.to_messages()


def test_instructions_are_returned_apart_from_the_messages() -> None:
    items, system = AnthropicMessageAdapter.to_messages(prompt=build_prompt())

    assert system == "You are terse."
    assert items == [{"role": "user", "content": "Capital of France?"}]


def test_several_instructions_are_joined_by_a_blank_line() -> None:
    prompt = RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content="Be terse."),
            RenderedPromptInstruction(
                role=MessageRole.DEVELOPER, content="Answer in French."
            ),
        ]
    )

    _, system = AnthropicMessageAdapter.to_messages(prompt=prompt)

    assert system == "Be terse.\n\nAnswer in French."


@pytest.mark.parametrize("role", [MessageRole.SYSTEM, MessageRole.DEVELOPER])
def test_a_stored_instruction_is_lifted_out_of_the_conversation(
    role: MessageRole,
) -> None:
    items, system = AnthropicMessageAdapter.to_messages(
        messages=[stored_message(role, "Be terse."), stored_message()]
    )

    assert system == "Be terse."
    assert items == [{"role": "user", "content": "hello"}]


def test_a_prompt_instruction_precedes_a_stored_one() -> None:
    _, system = AnthropicMessageAdapter.to_messages(
        prompt=build_prompt("From the prompt."),
        messages=[stored_message(MessageRole.SYSTEM, "From the history.")],
    )

    assert system == "From the prompt.\n\nFrom the history."


def test_no_instructions_at_all_means_no_system_field() -> None:
    _, system = AnthropicMessageAdapter.to_messages(messages=[stored_message()])

    assert system is None


def test_a_tool_result_travels_as_a_block_inside_a_user_turn() -> None:
    items, _ = AnthropicMessageAdapter.to_messages(
        messages=[tool_result("toolu_0", "21", 3)]
    )

    assert items == [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_0", "content": "21"}
            ],
        }
    ]


def test_a_tool_call_round_trips_as_a_tool_use_block() -> None:
    items, _ = AnthropicMessageAdapter.to_messages(
        messages=[assistant_with_tool_calls("get_weather")]
    )

    assert items == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_0",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        }
    ]


def test_tool_call_arguments_are_decoded_from_their_stored_json() -> None:
    message = assistant_with_tool_calls("get_weather")
    message.tool_calls[0].function.arguments = ""

    items, _ = AnthropicMessageAdapter.to_messages(messages=[message])

    assert items[0]["content"][0]["input"] == {}


def test_arguments_that_are_not_json_are_refused() -> None:
    message = assistant_with_tool_calls("get_weather")
    message.tool_calls[0].function.arguments = "not json"

    with pytest.raises(ValueError, match="not valid JSON"):
        AnthropicMessageAdapter.to_messages(messages=[message])


def test_arguments_that_decode_to_something_other_than_an_object_are_refused() -> None:
    message = assistant_with_tool_calls("get_weather")
    message.tool_calls[0].function.arguments = "[1, 2]"

    with pytest.raises(ValueError, match="must decode to an object"):
        AnthropicMessageAdapter.to_messages(messages=[message])


def test_parallel_tool_results_merge_into_one_user_turn() -> None:
    items, _ = AnthropicMessageAdapter.to_messages(
        messages=[
            assistant_with_tool_calls("a", "b"),
            tool_result("toolu_0", "21", 3),
            tool_result("toolu_1", "rain", 4),
        ]
    )

    assert [item["role"] for item in items] == ["assistant", "user"]
    assert [block["tool_use_id"] for block in items[-1]["content"]] == [
        "toolu_0",
        "toolu_1",
    ]


def test_merging_turns_a_plain_string_into_a_text_block() -> None:
    items, _ = AnthropicMessageAdapter.to_messages(
        messages=[
            stored_message(MessageRole.USER, "one", 1),
            stored_message(MessageRole.USER, "two", 2),
        ]
    )

    assert items == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ],
        }
    ]


def test_merging_does_not_mutate_the_turn_it_merged_into() -> None:
    first = {"role": "user", "content": "one"}
    second = {"role": "user", "content": "two"}

    merged = AnthropicMessageAdapter._merge_consecutive([first, second])

    assert first["content"] == "one"
    assert len(merged[0]["content"]) == 2


def test_alternating_roles_are_left_alone() -> None:
    items, _ = AnthropicMessageAdapter.to_messages(
        messages=[
            stored_message(MessageRole.USER, "one", 1),
            stored_message(MessageRole.ASSISTANT, "two", 2),
            stored_message(MessageRole.USER, "three", 3),
        ]
    )

    assert [item["role"] for item in items] == ["user", "assistant", "user"]
    assert items[0]["content"] == "one"
    assert items[1]["content"] == [{"type": "text", "text": "two"}]
    assert items[2]["content"] == "three"


def test_an_assistant_turn_with_text_and_a_call_emits_both_blocks() -> None:
    message = assistant_with_tool_calls("get_weather")
    message.content = "Looking it up."

    items, _ = AnthropicMessageAdapter.to_messages(messages=[message])

    assert [block["type"] for block in items[0]["content"]] == ["text", "tool_use"]


def test_an_assistant_turn_with_neither_text_nor_calls_sends_no_blocks() -> None:
    message = assistant_with_tool_calls("get_weather")
    message.content = ""
    message.tool_calls = None

    items, _ = AnthropicMessageAdapter.to_messages(messages=[message])

    assert items == [{"role": "assistant", "content": []}]


def test_stored_text_is_stripped_of_invisible_characters() -> None:
    items, _ = AnthropicMessageAdapter.to_messages(
        messages=[stored_message(MessageRole.USER, TAGGED)]
    )

    assert items[0]["content"] == "hello"


def test_a_stored_instruction_is_stripped_too() -> None:
    _, system = AnthropicMessageAdapter.to_messages(
        messages=[stored_message(MessageRole.SYSTEM, TAGGED)]
    )

    assert system == "hello"


def test_a_rendered_prompt_is_trusted_and_left_alone() -> None:
    prompt = RenderedPrompt(
        messages=[RenderedPromptMessage(role=MessageRole.USER, content=TAGGED)]
    )

    items, _ = AnthropicMessageAdapter.to_messages(prompt=prompt)

    assert items[0]["content"] == TAGGED


def test_a_prompt_tool_message_needs_the_call_it_answers() -> None:
    prompt = RenderedPrompt(
        messages=[
            RenderedPromptMessage(
                role=MessageRole.TOOL, content="21", tool_call_id="toolu_0"
            )
        ]
    )
    prompt.messages[0].tool_call_id = None

    with pytest.raises(ValueError, match="missing tool_call_id"):
        AnthropicMessageAdapter.to_messages(prompt=prompt)


def test_a_stored_tool_message_needs_the_call_it_answers() -> None:
    message = tool_result("toolu_0", "21", 3)
    message.tool_call_id = None

    with pytest.raises(ValueError, match="missing tool_call_id"):
        AnthropicMessageAdapter.to_messages(messages=[message])


def test_an_unsupported_prompt_role_is_refused() -> None:
    prompt = RenderedPrompt(
        messages=[RenderedPromptMessage(role=MessageRole.USER, content="x")]
    )
    prompt.messages[0].role = MessageRole.SYSTEM

    with pytest.raises(ValueError, match="Unsupported prompt message role"):
        AnthropicMessageAdapter.to_messages(prompt=prompt)


def test_an_unsupported_stored_role_is_refused() -> None:
    message = stored_message()
    message.role = "unknown"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unsupported stored message role"):
        AnthropicMessageAdapter.to_messages(messages=[message])


def test_content_parts_become_anthropic_blocks() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[
            TextPart(text=TAGGED),
            ImagePart(url="https://x/y.png"),
            ImagePart(data="QUJD", media_type="image/png"),
            DocumentPart(file_id="file-1"),
        ],
    )

    items, _ = AnthropicMessageAdapter.to_messages(messages=[message])

    assert items[0]["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
        },
        {"type": "document", "source": {"type": "file", "file_id": "file-1"}},
    ]


def test_an_inline_document_travels_as_base64() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[
            DocumentPart(data="QUJD", media_type="application/pdf", filename="a.pdf")
        ],
    )

    items, _ = AnthropicMessageAdapter.to_messages(messages=[message])

    assert items[0]["content"][0]["source"] == {
        "type": "base64",
        "media_type": "application/pdf",
        "data": "QUJD",
    }


def test_tools_carry_their_schema_under_input_schema() -> None:
    tools = [build_tool("alpha"), build_tool("beta")]

    shaped = AnthropicMessageAdapter.to_tools(tools)

    assert [item["name"] for item in shaped] == ["alpha", "beta"]
    assert shaped[0]["input_schema"] == tools[0].json_schema
    assert "parameters" not in shaped[0]


def test_no_tools_shapes_to_no_declarations() -> None:
    assert AnthropicMessageAdapter.to_tools([]) == []


def test_unicode_and_emoji_survive_the_adapter() -> None:
    text = "héllo 🌍 — naïve café"

    items, _ = AnthropicMessageAdapter.to_messages(
        messages=[stored_message(MessageRole.USER, text)]
    )

    assert items[0]["content"] == text
