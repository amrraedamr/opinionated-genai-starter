"""Pins the request Anthropic would have been sent.

Two things here are recent and easy to regress. Reasoning now travels as
`output_config.effort` rather than a `thinking.budget_tokens`, which the newer
models reject outright. And structured output is a declared JSON schema rather
than a tool the model is forced to call, so a parse reply arrives as text.

Caching is Anthropic's own mechanism: it caches nothing unless told where the
prefix ends, so the hint becomes a mark on the system block — added to the
text rather than replacing it, since a hinted call has to match the unhinted
ones before it.
"""

from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.definitions.model import CachePersistence, ReasoningEffort
from base.models.view_models import BaseModel
from base.models.view_models.exceptions.llm import LLMConfigError
from base.models.view_models.model import CacheHint, ModelConfigurations
from base.models.view_models.prompt import RenderedPrompt, RenderedPromptInstruction
from base.services.llm.anthropic import AnthropicService
from tests.fakes import build_prompt, build_tool, stored_message


class Answer(BaseModel):
    """The target of a parse call."""

    text: str


def bare_service() -> AnthropicService:
    """A service with only the attributes the shaping methods read."""
    service = object.__new__(AnthropicService)
    service._model = "claude-test"
    service._auth = None
    return service


def completion_request(**given: Any) -> dict[str, Any]:
    """The request a completion would have sent."""
    arguments: dict[str, Any] = {
        "prompt": build_prompt(),
        "messages": None,
        "tools": None,
        "max_output_tokens": None,
        "reasoning_effort": None,
        "cache_hint": None,
    }
    arguments.update(given)
    kwargs = arguments.pop("kwargs", {})
    _, request = bare_service()._prepare_completion(*arguments.values(), **kwargs)
    return request


def parse_request(**given: Any) -> dict[str, Any]:
    """The request a parse call would have sent."""
    arguments: dict[str, Any] = {
        "schema": Answer,
        "prompt": build_prompt(),
        "messages": None,
        "max_output_tokens": None,
        "reasoning_effort": None,
        "cache_hint": None,
    }
    arguments.update(given)
    kwargs = arguments.pop("kwargs", {})
    _, request = bare_service()._prepare_parse(*arguments.values(), **kwargs)
    return request


def test_the_effort_travels_as_an_output_config_rather_than_a_token_budget() -> None:
    request = completion_request(reasoning_effort=ReasoningEffort.HIGH)

    assert request["output_config"] == {"effort": "high"}
    assert "thinking" not in request


@pytest.mark.parametrize(
    "effort",
    [e for e in ReasoningEffort if e is not ReasoningEffort.NONE],
)
def test_every_effort_but_none_becomes_an_effort_word(effort: ReasoningEffort) -> None:
    request = completion_request(reasoning_effort=effort)

    assert request["output_config"]["effort"] == effort.value
    assert "budget_tokens" not in str(request)


def test_asking_for_no_reasoning_disables_thinking_outright() -> None:
    request = completion_request(reasoning_effort=ReasoningEffort.NONE)

    assert request["thinking"] == {"type": "disabled"}
    assert "output_config" not in request


def test_an_explicit_thinking_kwarg_is_left_alone() -> None:
    request = completion_request(
        reasoning_effort=ReasoningEffort.HIGH,
        kwargs={"thinking": {"type": "enabled", "budget_tokens": 2048}},
    )

    assert request["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in request


def test_an_explicit_output_config_effort_is_left_alone() -> None:
    request = completion_request(
        reasoning_effort=ReasoningEffort.LOW,
        kwargs={"output_config": {"effort": "max"}},
    )

    assert request["output_config"] == {"effort": "max"}


def test_no_effort_asked_for_means_no_reasoning_field_at_all() -> None:
    request = completion_request()

    assert "output_config" not in request
    assert "thinking" not in request


def test_a_missing_token_cap_falls_back_to_the_documented_default() -> None:
    assert completion_request()["max_tokens"] == (
        AnthropicService.DEFAULT_MAX_OUTPUT_TOKENS
    )
    assert completion_request(max_output_tokens=256)["max_tokens"] == 256


def test_instructions_leave_the_message_list_for_the_system_field() -> None:
    request = completion_request()

    assert request["system"] == "You are terse."
    assert request["messages"] == [{"role": "user", "content": "Capital of France?"}]


def test_tools_carry_their_schema_under_input_schema() -> None:
    tool = build_tool()
    request = completion_request(tools=[tool])

    assert request["tools"] == [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.json_schema,
        }
    ]


def test_structured_output_is_a_declared_schema_not_a_forced_tool() -> None:
    request = parse_request()

    assert request["output_config"]["format"] == {
        "type": "json_schema",
        "schema": Answer.model_json_schema(),
    }
    assert "tool_choice" not in request
    assert "tools" not in request


def test_a_parse_keeps_its_effort_alongside_its_format() -> None:
    request = parse_request(reasoning_effort=ReasoningEffort.MEDIUM)

    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"]["type"] == "json_schema"


def test_a_parse_does_not_mutate_a_caller_supplied_output_config() -> None:
    supplied: dict[str, Any] = {"effort": "low"}

    request = parse_request(kwargs={"output_config": supplied})

    assert supplied == {"effort": "low"}
    assert request["output_config"]["format"]["type"] == "json_schema"


def test_a_parse_schema_with_nested_models_is_sent_whole() -> None:
    class Inner(BaseModel):
        kind: str

    class Outer(BaseModel):
        inner: Inner

    request = parse_request(schema=Outer)

    assert "$defs" in request["output_config"]["format"]["schema"]


def test_no_cache_hint_leaves_the_system_block_as_plain_text() -> None:
    request = completion_request(cache_hint=None)

    assert request["system"] == "You are terse."


@pytest.mark.parametrize(
    ("persistence", "ttl"),
    [(CachePersistence.SHORT, "5m"), (CachePersistence.LONG, "1h")],
)
def test_each_persistence_level_becomes_a_duration(
    persistence: CachePersistence, ttl: str
) -> None:
    request = completion_request(cache_hint=CacheHint(persistence=persistence))

    assert request["system"] == [
        {
            "type": "text",
            "text": "You are terse.",
            "cache_control": {"type": "ephemeral", "ttl": ttl},
        }
    ]


def test_the_marked_text_is_the_unmarked_text_exactly() -> None:
    plain = completion_request(cache_hint=None)
    marked = completion_request(cache_hint=CacheHint())

    assert marked["system"][0]["text"] == plain["system"]


def test_marking_the_head_leaves_the_conversation_untouched() -> None:
    messages = [stored_message(MessageRole.USER, "one", 1)]
    plain = completion_request(messages=messages, cache_hint=None)
    marked = completion_request(messages=messages, cache_hint=CacheHint())

    assert marked["messages"] == plain["messages"]


def test_with_no_instructions_the_last_tool_carries_the_mark() -> None:
    tools = [build_tool("alpha"), build_tool("beta")]
    request = completion_request(
        prompt=RenderedPrompt(messages=build_prompt().messages),
        tools=tools,
        cache_hint=CacheHint(),
    )

    assert "system" not in request
    assert "cache_control" not in request["tools"][0]
    assert request["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_instructions_win_over_tools_as_the_place_to_mark() -> None:
    request = completion_request(tools=[build_tool()], cache_hint=CacheHint())

    assert request["system"][0]["cache_control"]["type"] == "ephemeral"
    assert "cache_control" not in request["tools"][-1]


def test_a_hint_with_neither_instructions_nor_tools_marks_nothing() -> None:
    request = completion_request(
        prompt=RenderedPrompt(messages=build_prompt().messages),
        cache_hint=CacheHint(),
    )

    assert "system" not in request
    assert "tools" not in request


def test_a_cache_key_is_meaningless_here_and_is_not_sent() -> None:
    request = completion_request(cache_hint=CacheHint(key="conversation-7"))

    assert "conversation-7" not in str(request)


def test_a_parse_marks_the_system_block_too() -> None:
    request = parse_request(cache_hint=CacheHint(persistence=CachePersistence.LONG))

    assert request["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_extra_kwargs_reach_the_request_unaltered() -> None:
    request = completion_request(kwargs={"top_k": 5, "stop_sequences": ["END"]})

    assert request["top_k"] == 5
    assert request["stop_sequences"] == ["END"]


def test_nothing_sets_a_temperature_unless_the_caller_does() -> None:
    assert "temperature" not in completion_request(
        reasoning_effort=ReasoningEffort.HIGH, max_output_tokens=256
    )
    assert completion_request(kwargs={"temperature": 0.3})["temperature"] == 0.3


def test_the_model_name_is_on_every_request() -> None:
    assert completion_request()["model"] == "claude-test"
    assert parse_request()["model"] == "claude-test"


def test_a_model_name_is_required() -> None:
    with pytest.raises(LLMConfigError, match="model_name is required"):
        AnthropicService(ModelConfigurations(api_host="http://x", api_key="k"))


def test_an_async_credential_refuses_the_sync_surface() -> None:
    service = bare_service()
    service._auth = object()

    with pytest.raises(LLMConfigError, match="no synchronous surface"):
        service._complete(build_prompt(), None, None, None, None, None, None, None)


def test_a_huge_instruction_is_sent_whole_and_marked_whole() -> None:
    instruction = "Follow the policy. " * 5000
    request = completion_request(
        prompt=RenderedPrompt(
            instructions=[
                RenderedPromptInstruction(role=MessageRole.SYSTEM, content=instruction)
            ],
            messages=build_prompt().messages,
        ),
        cache_hint=CacheHint(),
    )

    assert request["system"][0]["text"] == instruction
