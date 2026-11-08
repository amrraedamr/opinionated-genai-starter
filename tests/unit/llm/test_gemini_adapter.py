"""Pins how neutral messages become Gemini's contents.

Gemini differs in three ways the adapter has to absorb: the assistant's role
is called `model`, instructions travel on the request rather than in the
conversation, and a tool result is matched to its call by **function name**
rather than by an id. That last one goes both ways — an outgoing result
resolves its `tool_call_id` back to a name, and an incoming call that carries
no id is given an invented one.
"""

import json
import uuid
from typing import Any
from uuid import UUID

import pytest
from google.genai import types as genai_types

from base.models.definitions.message import MessageRole
from base.models.view_models import BaseModel
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
from base.services.llm.gemini import GeminiService
from base.services.llm.gemini.adapter import _ROLE_MAP, GeminiContentAdapter
from tests.fakes import build_prompt, build_tool, stored_message


def _service() -> GeminiService:
    bare = object.__new__(GeminiService)
    bare._model = "gemini-test"
    return bare


def _response(*parts: genai_types.Part) -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=list(parts)),
                finish_reason=genai_types.FinishReason.STOP,
            )
        ]
    )


CONVERSATION = uuid.UUID(int=1)

TAGGED = "hello\U000e0041\u200b"
"""Text carrying a tag character and a zero-width space."""


def assistant_with_tool_call(
    name: str = "get_weather", call_id: str = "call_1"
) -> StoredMessage:
    return StoredMessage(
        conversation_id=CONVERSATION,
        sequence=2,
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=[
            StoredToolCall(
                id=call_id,
                function=StoredToolCallFunction(
                    name=name, arguments='{"city": "Paris"}'
                ),
            )
        ],
    )


def tool_result(
    call_id: str = "call_1", content: str = '{"temp": 21}', sequence: int = 3
) -> StoredMessage:
    return StoredMessage(
        conversation_id=CONVERSATION,
        sequence=sequence,
        role=MessageRole.TOOL,
        content=content,
        tool_call_id=call_id,
    )


def test_an_adapter_call_with_neither_input_is_refused() -> None:
    with pytest.raises(ValueError, match="At least one of"):
        GeminiContentAdapter.to_contents()


def test_the_assistant_is_called_model() -> None:
    assert _ROLE_MAP[MessageRole.ASSISTANT] == "model"
    assert _ROLE_MAP[MessageRole.USER] == "user"
    assert _ROLE_MAP[MessageRole.TOOL] == "user"


def test_an_assistant_turn_arrives_under_the_model_role() -> None:
    contents, _ = GeminiContentAdapter.to_contents(
        messages=[stored_message(MessageRole.ASSISTANT, "Paris", 2)]
    )

    assert contents == [{"role": "model", "parts": [{"text": "Paris"}]}]


def test_a_prompt_assistant_message_arrives_under_the_model_role_too() -> None:
    prompt = RenderedPrompt(
        messages=[RenderedPromptMessage(role=MessageRole.ASSISTANT, content="Paris")]
    )

    contents, _ = GeminiContentAdapter.to_contents(prompt=prompt)

    assert contents == [{"role": "model", "parts": [{"text": "Paris"}]}]


def test_instructions_are_returned_apart_from_the_contents() -> None:
    contents, system = GeminiContentAdapter.to_contents(prompt=build_prompt())

    assert system == "You are terse."
    assert contents == [{"role": "user", "parts": [{"text": "Capital of France?"}]}]


def test_several_instructions_are_joined_by_a_blank_line() -> None:
    prompt = RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content="Be terse."),
            RenderedPromptInstruction(
                role=MessageRole.DEVELOPER, content="Answer in French."
            ),
        ]
    )

    _, system = GeminiContentAdapter.to_contents(prompt=prompt)

    assert system == "Be terse.\n\nAnswer in French."


@pytest.mark.parametrize("role", [MessageRole.SYSTEM, MessageRole.DEVELOPER])
def test_a_stored_instruction_is_lifted_out_of_the_conversation(
    role: MessageRole,
) -> None:
    contents, system = GeminiContentAdapter.to_contents(
        messages=[stored_message(role, TAGGED), stored_message()]
    )

    assert system == "hello"
    assert contents == [{"role": "user", "parts": [{"text": "hello"}]}]


def test_no_instructions_at_all_means_no_system_instruction() -> None:
    _, system = GeminiContentAdapter.to_contents(messages=[stored_message()])

    assert system is None


def test_a_tool_result_is_named_after_the_call_it_answers() -> None:
    contents, _ = GeminiContentAdapter.to_contents(
        messages=[assistant_with_tool_call("get_weather", "call_1"), tool_result()]
    )

    assert contents[-1] == {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "name": "get_weather",
                    "response": {"temp": 21},
                }
            }
        ],
    }


def test_a_tool_result_whose_call_is_absent_falls_back_to_its_id() -> None:
    contents, _ = GeminiContentAdapter.to_contents(messages=[tool_result("orphan_1")])

    assert contents[0]["parts"][0]["function_response"]["name"] == "orphan_1"


def test_a_tool_result_is_resolved_against_a_prompt_tool_call_too() -> None:
    prompt = RenderedPrompt(
        messages=[
            RenderedPromptMessage(
                role=MessageRole.TOOL, content='{"temp": 21}', tool_call_id="call_1"
            )
        ]
    )

    contents, _ = GeminiContentAdapter.to_contents(
        messages=[assistant_with_tool_call("get_weather", "call_1")], prompt=prompt
    )

    names = [
        part["function_response"]["name"]
        for item in contents
        for part in item["parts"]
        if "function_response" in part
    ]
    assert names == ["get_weather"]


def test_a_tool_result_that_is_not_json_is_wrapped_rather_than_dropped() -> None:
    contents, _ = GeminiContentAdapter.to_contents(
        messages=[tool_result("call_1", "21 degrees")]
    )

    assert contents[0]["parts"][0]["function_response"]["response"] == {
        "result": "21 degrees"
    }


def test_a_tool_result_that_decodes_to_a_scalar_is_wrapped_too() -> None:
    contents, _ = GeminiContentAdapter.to_contents(
        messages=[tool_result("call_1", "21")]
    )

    assert contents[0]["parts"][0]["function_response"]["response"] == {"result": 21}


def test_an_empty_tool_result_sends_an_empty_object() -> None:
    contents, _ = GeminiContentAdapter.to_contents(messages=[tool_result("call_1", "")])

    assert contents[0]["parts"][0]["function_response"]["response"] == {}


def test_an_outgoing_tool_call_carries_its_name_and_no_id() -> None:
    contents, _ = GeminiContentAdapter.to_contents(
        messages=[assistant_with_tool_call()]
    )

    assert contents[0]["parts"] == [
        {"function_call": {"name": "get_weather", "args": {"city": "Paris"}}}
    ]


def test_an_incoming_tool_call_without_an_id_is_given_an_invented_one() -> None:
    """The other direction of name-based matching: Gemini sends no call id."""

    class Call:
        name = "get_weather"
        args = {"city": "Paris"}

    class Part:
        function_call = Call()
        text = None

    class Content:
        parts = [Part()]

    class Candidate:
        content = Content()
        finish_reason = "STOP"

    class Response:
        candidates = [Candidate()]
        usage_metadata = None

    first = object.__new__(GeminiService)._to_result(Response())
    second = object.__new__(GeminiService)._to_result(Response())

    assert first.tool_calls is not None and second.tool_calls is not None
    assert first.tool_calls[0].function.name == "get_weather"
    assert uuid.UUID(first.tool_calls[0].id)
    assert first.tool_calls[0].id != second.tool_calls[0].id


def test_an_incoming_tool_call_keeps_an_id_when_one_arrives() -> None:
    class Call:
        id = "server_given"
        name = "get_weather"
        args = {}

    class Part:
        function_call = Call()
        text = None

    class Content:
        parts = [Part()]

    class Candidate:
        content = Content()
        finish_reason = "STOP"

    class Response:
        candidates = [Candidate()]
        usage_metadata = None

    result = object.__new__(GeminiService)._to_result(Response())

    assert result.tool_calls is not None
    assert result.tool_calls[0].id == "server_given"


def test_tool_call_arguments_that_are_not_json_become_an_empty_object() -> None:
    message = assistant_with_tool_call()
    message.tool_calls[0].function.arguments = "not json"

    contents, _ = GeminiContentAdapter.to_contents(messages=[message])

    assert contents[0]["parts"][0]["function_call"]["args"] == {}


def test_an_assistant_turn_with_text_and_a_call_emits_both_parts() -> None:
    message = assistant_with_tool_call()
    message.content = "Looking it up."

    contents, _ = GeminiContentAdapter.to_contents(messages=[message])

    assert contents[0]["parts"] == [
        {"text": "Looking it up."},
        {"function_call": {"name": "get_weather", "args": {"city": "Paris"}}},
    ]


def test_consecutive_same_role_turns_are_folded_together() -> None:
    contents, _ = GeminiContentAdapter.to_contents(
        messages=[
            stored_message(MessageRole.USER, "one", 1),
            stored_message(MessageRole.USER, "two", 2),
        ]
    )

    assert contents == [{"role": "user", "parts": [{"text": "one"}, {"text": "two"}]}]


def test_a_tool_result_folds_into_the_user_turn_before_it() -> None:
    contents, _ = GeminiContentAdapter.to_contents(
        messages=[
            assistant_with_tool_call("a", "call_1"),
            tool_result("call_1", "{}", 3),
            tool_result("call_1", "{}", 4),
        ]
    )

    assert [item["role"] for item in contents] == ["model", "user"]
    assert len(contents[-1]["parts"]) == 2


def test_alternating_roles_are_left_alone() -> None:
    contents, _ = GeminiContentAdapter.to_contents(
        messages=[
            stored_message(MessageRole.USER, "one", 1),
            stored_message(MessageRole.ASSISTANT, "two", 2),
            stored_message(MessageRole.USER, "three", 3),
        ]
    )

    assert [item["role"] for item in contents] == ["user", "model", "user"]


def test_stored_text_is_stripped_of_invisible_characters() -> None:
    contents, _ = GeminiContentAdapter.to_contents(
        messages=[stored_message(MessageRole.USER, TAGGED)]
    )

    assert contents[0]["parts"] == [{"text": "hello"}]


def test_a_rendered_prompt_is_trusted_and_left_alone() -> None:
    prompt = RenderedPrompt(
        messages=[RenderedPromptMessage(role=MessageRole.USER, content=TAGGED)]
    )

    contents, _ = GeminiContentAdapter.to_contents(prompt=prompt)

    assert contents[0]["parts"] == [{"text": TAGGED}]


def test_a_tool_message_needs_the_call_it_answers() -> None:
    message = tool_result()
    message.tool_call_id = None

    with pytest.raises(ValueError, match="missing tool_call_id"):
        GeminiContentAdapter.to_contents(messages=[message])


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
        GeminiContentAdapter.to_contents(prompt=prompt)


def test_an_unsupported_stored_role_is_refused() -> None:
    message = stored_message()
    message.role = "unknown"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unsupported stored message role"):
        GeminiContentAdapter.to_contents(messages=[message])


def test_an_unsupported_prompt_role_is_refused() -> None:
    prompt = RenderedPrompt(
        messages=[RenderedPromptMessage(role=MessageRole.USER, content="x")]
    )
    prompt.messages[0].role = MessageRole.SYSTEM

    with pytest.raises(ValueError, match="Unsupported prompt message role"):
        GeminiContentAdapter.to_contents(prompt=prompt)


def test_an_image_by_url_becomes_file_data() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[ImagePart(url="https://x/y.png", media_type="image/png")],
    )

    contents, _ = GeminiContentAdapter.to_contents(messages=[message])

    assert contents[0]["parts"] == [
        {"file_data": {"mime_type": "image/png", "file_uri": "https://x/y.png"}}
    ]


def test_inline_image_bytes_are_decoded_to_raw_bytes() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[ImagePart(data="QUJD", media_type="image/png")],
    )

    contents, _ = GeminiContentAdapter.to_contents(messages=[message])

    assert contents[0]["parts"] == [
        {"inline_data": {"mime_type": "image/png", "data": b"ABC"}}
    ]


def test_an_inline_document_defaults_to_pdf() -> None:
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[
            DocumentPart(data="QUJD", media_type="application/pdf", filename="a.pdf")
        ],
    )
    message.parts[0].media_type = None

    contents, _ = GeminiContentAdapter.to_contents(messages=[message])

    assert contents[0]["parts"][0]["inline_data"]["mime_type"] == "application/pdf"


def test_a_text_part_is_not_stripped_the_way_message_content_is() -> None:
    """Pins current behaviour: `to_parts` passes `TextPart.text` through."""
    message = StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        parts=[TextPart(text=TAGGED)],
    )

    contents, _ = GeminiContentAdapter.to_contents(messages=[message])

    assert contents[0]["parts"] == [{"text": TAGGED}]


def test_tools_are_gathered_into_one_declaration_list() -> None:
    shaped = GeminiContentAdapter.to_tools([build_tool("alpha"), build_tool("beta")])

    assert len(shaped) == 1
    assert [item["name"] for item in shaped[0]["function_declarations"]] == [
        "alpha",
        "beta",
    ]


def test_a_tool_schema_is_reduced_to_the_subset_gemini_accepts() -> None:
    shaped = GeminiContentAdapter.to_tools([build_tool()])
    parameters = shaped[0]["function_declarations"][0]["parameters"]

    assert "title" not in json.dumps(parameters)
    assert parameters["properties"]["city"] == {"type": "string"}


def test_a_reference_is_inlined_and_its_definitions_dropped() -> None:
    class Inner(BaseModel):
        kind: str

    class Outer(BaseModel):
        name: str
        inner: Inner

    shaped = GeminiContentAdapter.to_schema(Outer.model_json_schema())

    assert "$defs" not in shaped
    assert "$ref" not in json.dumps(shaped)
    assert shaped["properties"]["inner"] == {
        "properties": {"kind": {"type": "string"}},
        "required": ["kind"],
        "type": "object",
    }


def test_a_reference_used_twice_is_inlined_twice() -> None:
    class Inner(BaseModel):
        kind: str

    class Outer(BaseModel):
        first: Inner
        second: Inner

    shaped = GeminiContentAdapter.to_schema(Outer.model_json_schema())

    assert shaped["properties"]["first"] == shaped["properties"]["second"]
    assert shaped["properties"]["first"]["type"] == "object"


def test_a_reference_inside_a_list_is_inlined() -> None:
    class Inner(BaseModel):
        kind: str

    class Outer(BaseModel):
        many: list[Inner]

    shaped = GeminiContentAdapter.to_schema(Outer.model_json_schema())

    assert shaped["properties"]["many"]["items"]["type"] == "object"
    assert "$ref" not in json.dumps(shaped)


def test_additional_properties_and_schema_keywords_are_dropped() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "title": "X",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {"a": {"type": "string", "title": "A"}},
    }

    assert GeminiContentAdapter.to_schema(schema) == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }


def test_an_unknown_reference_resolves_to_an_empty_object() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Missing"}},
    }

    assert GeminiContentAdapter.to_schema(schema) == {
        "type": "object",
        "properties": {"a": {}},
    }


def test_unicode_and_emoji_survive_the_adapter() -> None:
    text = "héllo 🌍 — naïve café"

    contents, _ = GeminiContentAdapter.to_contents(
        messages=[stored_message(MessageRole.USER, text)]
    )

    assert contents[0]["parts"] == [{"text": text}]


# Gemini 3.x refuses a replayed function call that carries no thought
# signature: "Function call is missing a thought_signature in functionCall
# parts. This is required for tools to work correctly." Nothing in the
# neutral shape kept one, so a second turn could never be sent.


def test_a_tool_call_keeps_the_signature_the_provider_issued() -> None:
    """Read off the reply, so it can be handed back on the next turn."""
    part = genai_types.Part(
        function_call=genai_types.FunctionCall(
            name="get_weather", args={"city": "Oslo"}
        )
    )
    part.thought_signature = b"opaque-token"

    result = GeminiService._to_result(_service(), _response(part))

    assert result.tool_calls
    assert result.tool_calls[0].provider_signature == "b3BhcXVlLXRva2Vu"


def test_a_tool_call_without_a_signature_keeps_none() -> None:
    part = genai_types.Part(
        function_call=genai_types.FunctionCall(name="get_weather", args={})
    )
    result = GeminiService._to_result(_service(), _response(part))

    assert result.tool_calls
    assert result.tool_calls[0].provider_signature is None


def test_a_replayed_call_carries_its_signature_back() -> None:
    """The part sent on the next turn has to repeat what the model issued."""
    message = StoredMessage(
        conversation_id=UUID(int=3),
        sequence=1,
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            StoredToolCall(
                id="c1",
                function=StoredToolCallFunction(
                    name="get_weather", arguments='{"city": "Oslo"}'
                ),
                provider_signature="b3BhcXVlLXRva2Vu",
            )
        ],
    )

    wire, _ = GeminiContentAdapter.to_contents(messages=[message])

    part = wire[0]["parts"][0]
    assert part["function_call"]["name"] == "get_weather"
    assert part["thought_signature"] == "b3BhcXVlLXRva2Vu"


def test_a_replayed_call_without_a_signature_sends_no_field() -> None:
    """An older model issues none, and must not be sent an empty one."""
    message = StoredMessage(
        conversation_id=UUID(int=3),
        sequence=1,
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            StoredToolCall(
                id="c1",
                function=StoredToolCallFunction(name="get_weather", arguments="{}"),
            )
        ],
    )

    wire, _ = GeminiContentAdapter.to_contents(messages=[message])

    assert "thought_signature" not in wire[0]["parts"][0]
