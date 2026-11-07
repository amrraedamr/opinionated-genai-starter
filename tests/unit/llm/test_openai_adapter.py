"""Pins how neutral messages become OpenAI's two wire shapes.

Chat Completions keeps a tool result in its own role and nests tool calls on
the assistant message; the Responses API flattens both into items of their
own. Both strip invisible characters from stored text, which arrives from
outside the application, and neither strips them from a rendered prompt, which
does not.
"""

import uuid
from typing import Any

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
from base.services.llm.openai.adapter import OpenAIMessageAdapter
from tests.fakes import build_prompt, build_tool, stored_message

CONVERSATION = uuid.UUID(int=1)

TAGGED = "hello\U000e0041\u200b"
"""Text carrying a tag character and a zero-width space."""


def assistant_with_tool_call(name: str = "get_weather") -> StoredMessage:
    """An assistant turn that asked for a tool and said nothing else."""
    return StoredMessage(
        conversation_id=CONVERSATION,
        sequence=2,
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=[
            StoredToolCall(
                id="call_1",
                function=StoredToolCallFunction(
                    name=name, arguments='{"city": "Paris"}'
                ),
            )
        ],
    )


def tool_result(content: str = "21") -> StoredMessage:
    """The reply to `assistant_with_tool_call`."""
    return StoredMessage(
        conversation_id=CONVERSATION,
        sequence=3,
        role=MessageRole.TOOL,
        content=content,
        tool_call_id="call_1",
    )


def test_an_adapter_call_with_neither_input_is_refused() -> None:
    with pytest.raises(ValueError, match="At least one of"):
        OpenAIMessageAdapter.to_chat_completions_messages()
    with pytest.raises(ValueError, match="At least one of"):
        OpenAIMessageAdapter.to_responses_input()


def test_chat_completions_keeps_a_prompt_in_order() -> None:
    result = OpenAIMessageAdapter.to_chat_completions_messages(prompt=build_prompt())

    assert result == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Capital of France?"},
    ]


def test_a_developer_instruction_keeps_its_own_role() -> None:
    prompt = RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.DEVELOPER, content="Be terse.")
        ]
    )

    assert OpenAIMessageAdapter.to_chat_completions_messages(prompt=prompt) == [
        {"role": "developer", "content": "Be terse."}
    ]
    assert OpenAIMessageAdapter.to_responses_input(prompt=prompt) == [
        {"role": "developer", "content": "Be terse."}
    ]


def test_a_prompt_tool_message_needs_the_call_it_answers() -> None:
    prompt = RenderedPrompt(
        messages=[
            RenderedPromptMessage(
                role=MessageRole.TOOL, content="21", tool_call_id="call_1"
            )
        ]
    )
    prompt.messages[0].tool_call_id = None

    with pytest.raises(ValueError, match="missing tool_call_id"):
        OpenAIMessageAdapter.to_chat_completions_messages(prompt=prompt)
    with pytest.raises(ValueError, match="missing tool_call_id"):
        OpenAIMessageAdapter.to_responses_input(prompt=prompt)


def test_chat_completions_nests_a_tool_call_under_the_assistant_turn() -> None:
    result = OpenAIMessageAdapter.to_chat_completions_messages(
        messages=[assistant_with_tool_call()]
    )

    assert result == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    ]


def test_responses_lifts_a_tool_call_into_an_item_of_its_own() -> None:
    result = OpenAIMessageAdapter.to_responses_input(
        messages=[assistant_with_tool_call()]
    )

    assert result == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        }
    ]


def test_responses_drops_the_empty_text_of_a_tool_only_assistant_turn() -> None:
    message = assistant_with_tool_call()
    message.content = ""

    result = OpenAIMessageAdapter.to_responses_input(messages=[message])

    assert [item["type"] for item in result] == ["function_call"]


def test_an_assistant_turn_with_both_text_and_a_call_emits_both_items() -> None:
    message = assistant_with_tool_call()
    message.content = "Looking it up."

    result = OpenAIMessageAdapter.to_responses_input(messages=[message])

    assert [item.get("type") or item["role"] for item in result] == [
        "assistant",
        "function_call",
    ]


def test_a_tool_result_round_trips_through_both_surfaces() -> None:
    completions = OpenAIMessageAdapter.to_chat_completions_messages(
        messages=[assistant_with_tool_call(), tool_result()]
    )
    responses = OpenAIMessageAdapter.to_responses_input(
        messages=[assistant_with_tool_call(), tool_result()]
    )

    assert completions[-1] == {
        "role": "tool",
        "content": "21",
        "tool_call_id": "call_1",
    }
    assert responses[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "21",
    }


def test_a_stored_tool_message_without_its_id_is_refused() -> None:
    message = tool_result()
    message.tool_call_id = None

    with pytest.raises(ValueError, match="missing tool_call_id"):
        OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])
    with pytest.raises(ValueError, match="missing tool_call_id"):
        OpenAIMessageAdapter.to_responses_input(messages=[message])


@pytest.mark.parametrize("role", [MessageRole.SYSTEM, MessageRole.DEVELOPER])
def test_a_stored_instruction_keeps_its_role_on_both_surfaces(
    role: MessageRole,
) -> None:
    message = stored_message(role, "Be terse.")

    assert OpenAIMessageAdapter.to_chat_completions_messages(messages=[message]) == [
        {"role": role.value, "content": "Be terse."}
    ]
    assert OpenAIMessageAdapter.to_responses_input(messages=[message]) == [
        {"role": role.value, "content": "Be terse."}
    ]


def test_stored_text_is_stripped_of_invisible_characters() -> None:
    message = stored_message(MessageRole.USER, TAGGED)

    completions = OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])
    responses = OpenAIMessageAdapter.to_responses_input(messages=[message])

    assert completions[0]["content"] == "hello"
    assert responses[0]["content"] == "hello"


def test_a_rendered_prompt_is_trusted_and_left_alone() -> None:
    prompt = RenderedPrompt(
        messages=[RenderedPromptMessage(role=MessageRole.USER, content=TAGGED)]
    )

    result = OpenAIMessageAdapter.to_chat_completions_messages(prompt=prompt)

    assert result[0]["content"] == TAGGED


def test_stripping_reaches_assistant_text_too() -> None:
    message = stored_message(MessageRole.ASSISTANT, TAGGED)

    result = OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])

    assert result[0]["content"] == "hello"


def test_an_assistant_turn_with_no_text_keeps_its_none_rather_than_becoming_empty() -> (
    None
):
    result = OpenAIMessageAdapter.to_chat_completions_messages(
        messages=[assistant_with_tool_call()]
    )

    assert result[0]["content"] is None


def test_an_empty_prompt_message_becomes_empty_text_not_none() -> None:
    prompt = RenderedPrompt(
        messages=[RenderedPromptMessage(role=MessageRole.USER, content="")]
    )

    assert OpenAIMessageAdapter.to_chat_completions_messages(prompt=prompt) == [
        {"role": "user", "content": ""}
    ]


def test_prompt_items_precede_stored_items() -> None:
    result = OpenAIMessageAdapter.to_chat_completions_messages(
        prompt=build_prompt(),
        messages=[stored_message(MessageRole.USER, "later", 9)],
    )

    assert [item["content"] for item in result] == [
        "You are terse.",
        "Capital of France?",
        "later",
    ]


def test_an_unsupported_prompt_instruction_role_is_refused() -> None:
    instruction = RenderedPromptInstruction(
        role=MessageRole.SYSTEM, content="Be terse."
    )
    instruction.role = MessageRole.USER

    with pytest.raises(ValueError, match="Unsupported prompt instruction role"):
        OpenAIMessageAdapter.to_chat_completions_messages(
            prompt=RenderedPrompt(instructions=[instruction])
        )
    with pytest.raises(ValueError, match="Unsupported prompt instruction role"):
        OpenAIMessageAdapter.to_responses_input(
            prompt=RenderedPrompt(instructions=[instruction])
        )


def test_an_unsupported_stored_role_is_refused() -> None:
    message = stored_message()
    message.role = "unknown"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unsupported stored message role"):
        OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])
    with pytest.raises(ValueError, match="Unsupported stored message role"):
        OpenAIMessageAdapter.to_responses_input(messages=[message])


def test_text_parts_are_stripped_and_named_per_surface() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[TextPart(text=TAGGED)],
    )

    completions = OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])
    responses = OpenAIMessageAdapter.to_responses_input(messages=[message])

    assert completions[0]["content"] == [{"type": "text", "text": "hello"}]
    assert responses[0]["content"] == [{"type": "input_text", "text": "hello"}]


def test_an_image_part_by_url_is_named_per_surface() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[ImagePart(url="https://x/y.png", detail="high")],
    )

    completions = OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])
    responses = OpenAIMessageAdapter.to_responses_input(messages=[message])

    assert completions[0]["content"] == [
        {"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "high"}}
    ]
    assert responses[0]["content"] == [
        {"type": "input_image", "image_url": "https://x/y.png", "detail": "high"}
    ]


def test_inline_image_bytes_become_a_data_url() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[ImagePart(data="QUJD", media_type="image/png")],
    )

    result = OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])

    assert result[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_a_document_by_id_is_sent_by_id_rather_than_by_bytes() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[DocumentPart(file_id="file-1")],
    )

    completions = OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])
    responses = OpenAIMessageAdapter.to_responses_input(messages=[message])

    assert completions[0]["content"] == [
        {"type": "file", "file": {"file_id": "file-1"}}
    ]
    assert responses[0]["content"] == [{"type": "input_file", "file_id": "file-1"}]


def test_an_inline_document_carries_its_filename_and_bytes() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[
            DocumentPart(data="QUJD", media_type="application/pdf", filename="a.pdf")
        ],
    )

    result = OpenAIMessageAdapter.to_responses_input(messages=[message])

    assert result[0]["content"] == [
        {
            "type": "input_file",
            "filename": "a.pdf",
            "file_data": "data:application/pdf;base64,QUJD",
        }
    ]


def test_parts_win_over_content_on_a_user_turn() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        content="ignored",
        parts=[TextPart(text="used")],
    )

    result = OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])

    assert result[0]["content"] == [{"type": "text", "text": "used"}]


def test_chat_completions_nests_a_tool_declaration_under_function() -> None:
    tool = build_tool()

    assert OpenAIMessageAdapter.to_chat_completions_tools([tool]) == [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.json_schema,
            },
        }
    ]


def test_responses_flattens_a_tool_declaration() -> None:
    tool = build_tool()

    assert OpenAIMessageAdapter.to_responses_tools([tool]) == [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.json_schema,
        }
    ]


def test_tool_declarations_keep_the_order_they_were_given() -> None:
    tools = [build_tool("alpha"), build_tool("beta")]

    for shaped in (
        OpenAIMessageAdapter.to_responses_tools(tools),
        OpenAIMessageAdapter.to_chat_completions_tools(tools),
    ):
        names = [item.get("name") or item["function"]["name"] for item in shaped]
        assert names == ["alpha", "beta"]


def test_no_tools_shapes_to_no_declarations() -> None:
    assert OpenAIMessageAdapter.to_responses_tools([]) == []
    assert OpenAIMessageAdapter.to_chat_completions_tools([]) == []


def test_unicode_and_emoji_survive_both_surfaces() -> None:
    text = "héllo 🌍 — naïve café"
    message = stored_message(MessageRole.USER, text)

    completions: list[Any] = OpenAIMessageAdapter.to_chat_completions_messages(
        messages=[message]
    )
    responses: list[Any] = OpenAIMessageAdapter.to_responses_input(messages=[message])

    assert completions[0]["content"] == text
    assert responses[0]["content"] == text
