"""Pins the awkward shapes a caller can hand all three adapters at once.

Each case is run through every provider rather than one, because the point is
that the neutral seam absorbs the difference: a prompt with no instructions, a
prompt with nothing but instructions, a conversation and a prompt together,
tools alongside structured output, and content the wire format has to carry
intact — emoji, combining marks and a schema that nests models.
"""

from collections.abc import Callable
from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.definitions.model import ReasoningEffort
from base.models.view_models import BaseModel
from base.models.view_models.model import CacheHint
from base.models.view_models.prompt import (
    RenderedPrompt,
    RenderedPromptInstruction,
    RenderedPromptMessage,
)
from base.services.llm.anthropic import AnthropicService
from base.services.llm.anthropic.adapter import AnthropicMessageAdapter
from base.services.llm.gemini import GeminiService
from base.services.llm.gemini.adapter import GeminiContentAdapter
from base.services.llm.openai import OpenAIService
from base.services.llm.openai.adapter import OpenAIMessageAdapter
from tests.fakes import build_prompt, build_tool, stored_message

AWKWARD = "héllo 🌍 — e\u0301le\u0300ve ☃️ 日本語 \U0001fae0"
"""Emoji with a modifier, combining accents, CJK and an astral codepoint."""


class Inner(BaseModel):
    """A model nested inside another, so the schema carries a reference."""

    kind: str
    weight: float = 0.0


class Nested(BaseModel):
    """A schema with a `$ref`, a list of refs and a default."""

    name: str
    primary: Inner
    others: list[Inner] = []


def nesting_tool() -> Any:
    """A tool whose argument model nests another model."""
    from base.services.tool_calling import Tool

    async def handler(args: Nested) -> str:
        return args.name

    return Tool(
        name="describe",
        description="Describe a nested thing.",
        parameters=Nested,
        handler=handler,
    )


def anthropic_request(**given: Any) -> dict[str, Any]:
    service = object.__new__(AnthropicService)
    service._model = "claude-test"
    _, request = service._prepare_completion(
        given.get("prompt"),
        given.get("messages"),
        given.get("tools"),
        given.get("max_output_tokens"),
        given.get("reasoning_effort"),
        given.get("cache_hint"),
    )
    return request


def gemini_request(**given: Any) -> dict[str, Any]:
    service = object.__new__(GeminiService)
    service._model = "gemini-test"
    _, request = service._prepare_completion(
        given.get("prompt"),
        given.get("messages"),
        given.get("tools"),
        given.get("max_output_tokens"),
        given.get("reasoning_effort"),
        given.get("cache_hint"),
    )
    return request


SHAPERS: dict[str, Callable[..., dict[str, Any]]] = {
    "anthropic": anthropic_request,
    "gemini": gemini_request,
}


def instruction_only_prompt(content: str = "Be terse.") -> RenderedPrompt:
    return RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content=content)
        ]
    )


def message_only_prompt(content: str = "Capital of France?") -> RenderedPrompt:
    return RenderedPrompt(
        messages=[RenderedPromptMessage(role=MessageRole.USER, content=content)]
    )


def test_a_prompt_with_no_instructions_reaches_every_adapter_intact() -> None:
    prompt = message_only_prompt()

    completions = OpenAIMessageAdapter.to_chat_completions_messages(prompt=prompt)
    responses = OpenAIMessageAdapter.to_responses_input(prompt=prompt)
    anthropic_items, anthropic_system = AnthropicMessageAdapter.to_messages(
        prompt=prompt
    )
    gemini_items, gemini_system = GeminiContentAdapter.to_contents(prompt=prompt)

    assert completions == [{"role": "user", "content": "Capital of France?"}]
    assert responses == [{"role": "user", "content": "Capital of France?"}]
    assert anthropic_system is None
    assert anthropic_items == [{"role": "user", "content": "Capital of France?"}]
    assert gemini_system is None
    assert gemini_items == [{"role": "user", "parts": [{"text": "Capital of France?"}]}]


def test_a_prompt_with_only_instructions_reaches_every_adapter_intact() -> None:
    prompt = instruction_only_prompt()

    completions = OpenAIMessageAdapter.to_chat_completions_messages(prompt=prompt)
    anthropic_items, anthropic_system = AnthropicMessageAdapter.to_messages(
        prompt=prompt
    )
    gemini_items, gemini_system = GeminiContentAdapter.to_contents(prompt=prompt)

    assert completions == [{"role": "system", "content": "Be terse."}]
    assert (anthropic_system, anthropic_items) == ("Be terse.", [])
    assert (gemini_system, gemini_items) == ("Be terse.", [])


def test_a_prompt_and_messages_together_keep_the_prompt_in_front() -> None:
    prompt = build_prompt()
    messages = [stored_message(MessageRole.USER, "from history", 9)]

    completions = OpenAIMessageAdapter.to_chat_completions_messages(
        prompt=prompt, messages=messages
    )
    anthropic_items, _ = AnthropicMessageAdapter.to_messages(
        prompt=prompt, messages=messages
    )
    gemini_items, _ = GeminiContentAdapter.to_contents(prompt=prompt, messages=messages)

    assert [item["content"] for item in completions] == [
        "You are terse.",
        "Capital of France?",
        "from history",
    ]
    assert anthropic_items[0]["content"] == [
        {"type": "text", "text": "Capital of France?"},
        {"type": "text", "text": "from history"},
    ]
    assert gemini_items[0]["parts"] == [
        {"text": "Capital of France?"},
        {"text": "from history"},
    ]


def test_instructions_from_a_prompt_and_from_history_are_both_kept() -> None:
    prompt = instruction_only_prompt("From the prompt.")
    messages = [stored_message(MessageRole.DEVELOPER, "From the history.")]

    _, anthropic_system = AnthropicMessageAdapter.to_messages(
        prompt=prompt, messages=messages
    )
    _, gemini_system = GeminiContentAdapter.to_contents(
        prompt=prompt, messages=messages
    )
    completions = OpenAIMessageAdapter.to_chat_completions_messages(
        prompt=prompt, messages=messages
    )

    assert anthropic_system == "From the prompt.\n\nFrom the history."
    assert gemini_system == "From the prompt.\n\nFrom the history."
    assert [item["role"] for item in completions] == ["system", "developer"]


@pytest.mark.parametrize("effort", list(ReasoningEffort))
def test_every_effort_is_accepted_by_every_provider(effort: ReasoningEffort) -> None:
    openai_kwargs: dict[str, Any] = {}
    openai = object.__new__(OpenAIService)
    openai._supports_responses_api = True
    openai._apply_generation_knobs(openai_kwargs, None, effort)

    anthropic = anthropic_request(prompt=build_prompt(), reasoning_effort=effort)
    gemini = gemini_request(prompt=build_prompt(), reasoning_effort=effort)

    assert openai_kwargs["reasoning"]["effort"] is effort
    assert ("thinking" in anthropic) is (effort is ReasoningEffort.NONE)
    assert ("thinking_config" in gemini["config"]) is (
        effort is not ReasoningEffort.NONE
    )


def test_the_maximum_effort_reaches_every_provider_as_its_own_word() -> None:
    openai_kwargs: dict[str, Any] = {}
    openai = object.__new__(OpenAIService)
    openai._supports_responses_api = False
    openai._apply_generation_knobs(openai_kwargs, None, ReasoningEffort.MAX)

    assert openai_kwargs["reasoning_effort"] is ReasoningEffort.MAX
    assert anthropic_request(
        prompt=build_prompt(), reasoning_effort=ReasoningEffort.MAX
    )["output_config"] == {"effort": "max"}
    assert (
        gemini_request(prompt=build_prompt(), reasoning_effort=ReasoningEffort.MAX)[
            "config"
        ]["thinking_config"]["thinking_budget"]
        == (GeminiService.DEFAULT_THINKING_BUDGETS[ReasoningEffort.MAX])
    )


def test_tools_and_a_token_cap_travel_together_for_every_provider() -> None:
    tools = [build_tool("alpha"), build_tool("beta")]

    anthropic = anthropic_request(
        prompt=build_prompt(), tools=tools, max_output_tokens=256
    )
    gemini = gemini_request(prompt=build_prompt(), tools=tools, max_output_tokens=256)

    assert [item["name"] for item in anthropic["tools"]] == ["alpha", "beta"]
    assert anthropic["max_tokens"] == 256
    assert gemini["config"]["max_output_tokens"] == 256
    assert len(gemini["config"]["tools"][0]["function_declarations"]) == 2


def test_structured_output_and_a_cache_hint_coexist_on_anthropic() -> None:
    service = object.__new__(AnthropicService)
    service._model = "claude-test"

    _, request = service._prepare_parse(
        Nested, build_prompt(), None, 256, ReasoningEffort.HIGH, CacheHint()
    )

    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["effort"] == "high"
    assert request["system"][0]["cache_control"]["ttl"] == "5m"


def test_structured_output_and_a_cache_hint_coexist_on_openai() -> None:
    service = object.__new__(OpenAIService)
    service._supports_responses_api = True
    request: dict[str, Any] = {}

    service._apply_cache_hint(request, CacheHint(), build_prompt(), schema=Nested)

    assert request["prompt_cache_key"] == OpenAIService._head_fingerprint(
        build_prompt(), None, Nested, None
    )


def test_a_tool_whose_schema_nests_models_survives_every_adapter() -> None:
    tool = nesting_tool()

    responses = OpenAIMessageAdapter.to_responses_tools([tool])
    completions = OpenAIMessageAdapter.to_chat_completions_tools([tool])
    anthropic = AnthropicMessageAdapter.to_tools([tool])
    gemini = GeminiContentAdapter.to_tools([tool])

    assert "$defs" in responses[0]["parameters"]
    assert "$defs" in completions[0]["function"]["parameters"]
    assert "$defs" in anthropic[0]["input_schema"]
    declaration = gemini[0]["function_declarations"][0]
    assert "$defs" not in declaration["parameters"]
    assert declaration["parameters"]["properties"]["primary"]["type"] == "object"
    assert (
        declaration["parameters"]["properties"]["others"]["items"]["type"] == "object"
    )


def test_a_nested_schema_keeps_its_defaults_through_gemini() -> None:
    shaped = GeminiContentAdapter.to_schema(Nested.model_json_schema())

    assert shaped["properties"]["others"]["default"] == []
    assert shaped["properties"]["primary"]["properties"]["weight"]["default"] == 0.0


def test_awkward_unicode_survives_every_adapter() -> None:
    message = stored_message(MessageRole.USER, AWKWARD)

    completions = OpenAIMessageAdapter.to_chat_completions_messages(messages=[message])
    responses = OpenAIMessageAdapter.to_responses_input(messages=[message])
    anthropic_items, _ = AnthropicMessageAdapter.to_messages(messages=[message])
    gemini_items, _ = GeminiContentAdapter.to_contents(messages=[message])

    assert completions[0]["content"] == AWKWARD
    assert responses[0]["content"] == AWKWARD
    assert anthropic_items[0]["content"] == AWKWARD
    assert gemini_items[0]["parts"] == [{"text": AWKWARD}]


def test_awkward_unicode_survives_an_instruction() -> None:
    prompt = instruction_only_prompt(AWKWARD)

    _, anthropic_system = AnthropicMessageAdapter.to_messages(prompt=prompt)
    _, gemini_system = GeminiContentAdapter.to_contents(prompt=prompt)

    assert anthropic_system == AWKWARD
    assert gemini_system == AWKWARD


def test_a_huge_instruction_is_neither_truncated_nor_rewritten() -> None:
    instruction = "Follow the policy carefully. " * 20_000
    prompt = RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content=instruction)
        ],
        messages=build_prompt().messages,
    )

    anthropic = anthropic_request(prompt=prompt, cache_hint=CacheHint())
    gemini = gemini_request(prompt=prompt)
    completions = OpenAIMessageAdapter.to_chat_completions_messages(prompt=prompt)

    assert anthropic["system"][0]["text"] == instruction
    assert gemini["config"]["system_instruction"] == instruction
    assert completions[0]["content"] == instruction


def test_a_huge_instruction_fingerprints_to_the_usual_short_key() -> None:
    prompt = instruction_only_prompt("Follow the policy. " * 20_000)

    assert len(OpenAIService._head_fingerprint(prompt, None)) == 32


@pytest.mark.parametrize("shaper", list(SHAPERS.values()), ids=list(SHAPERS))
def test_messages_alone_shape_a_request_for_every_provider(
    shaper: Callable[..., dict[str, Any]],
) -> None:
    request = shaper(messages=[stored_message(MessageRole.USER, "Hello")])

    assert "Hello" in str(request)


@pytest.mark.parametrize("shaper", list(SHAPERS.values()), ids=list(SHAPERS))
def test_an_instruction_only_call_shapes_a_request_for_every_provider(
    shaper: Callable[..., dict[str, Any]],
) -> None:
    request = shaper(prompt=instruction_only_prompt())

    assert "Be terse." in str(request)


def test_an_empty_assistant_turn_does_not_become_a_blank_message() -> None:
    message = stored_message(MessageRole.ASSISTANT, "", 2)

    responses = OpenAIMessageAdapter.to_responses_input(messages=[message])
    anthropic_items, _ = AnthropicMessageAdapter.to_messages(messages=[message])
    gemini_items, _ = GeminiContentAdapter.to_contents(messages=[message])

    assert responses == []
    assert anthropic_items == [{"role": "assistant", "content": []}]
    assert gemini_items == [{"role": "model", "parts": []}]


def test_a_conversation_of_only_instructions_leaves_no_turns_behind() -> None:
    messages = [
        stored_message(MessageRole.SYSTEM, "One.", 1),
        stored_message(MessageRole.DEVELOPER, "Two.", 2),
    ]

    anthropic_items, anthropic_system = AnthropicMessageAdapter.to_messages(
        messages=messages
    )
    gemini_items, gemini_system = GeminiContentAdapter.to_contents(messages=messages)

    assert anthropic_items == []
    assert gemini_items == []
    assert anthropic_system == gemini_system == "One.\n\nTwo."
