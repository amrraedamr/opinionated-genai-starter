"""Pins the config Gemini would have been sent, and the hint it ignores.

`_apply_cache_hint` is deliberately empty here: Gemini caches a repeated prefix
by itself and offers no routing key, and its explicit caching is a resource
with a lifetime rather than a per-call flag. A caller holding one still has a
way through, because `**kwargs` reaches the config unaltered — so the test for
"adds nothing" and the test for "`cached_content` still arrives" are the same
claim seen from both sides.
"""

from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.definitions.model import CachePersistence, ReasoningEffort
from base.models.view_models import BaseModel
from base.models.view_models.exceptions.llm import LLMConfigError
from base.models.view_models.model import CacheHint, ModelConfigurations
from base.models.view_models.prompt import RenderedPrompt, RenderedPromptInstruction
from base.services.llm.gemini import GeminiService
from tests.fakes import build_prompt, build_tool, stored_message


class Answer(BaseModel):
    """The target of a parse call."""

    text: str


def bare_service() -> GeminiService:
    """A service with only the attributes the shaping methods read."""
    service = object.__new__(GeminiService)
    service._model = "gemini-test"
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


@pytest.mark.parametrize("persistence", list(CachePersistence))
def test_a_cache_hint_adds_nothing_to_the_config(
    persistence: CachePersistence,
) -> None:
    config: dict[str, Any] = {"system_instruction": "Be terse."}

    bare_service()._apply_cache_hint(
        config,
        CacheHint(persistence=persistence, key="conversation-7"),
        build_prompt(),
        messages=[stored_message()],
        tools=[build_tool()],
        schema=Answer,
    )

    assert config == {"system_instruction": "Be terse."}


def test_a_hinted_completion_is_identical_to_an_unhinted_one() -> None:
    assert completion_request(cache_hint=CacheHint()) == completion_request(
        cache_hint=None
    )


def test_a_hinted_parse_is_identical_to_an_unhinted_one() -> None:
    assert parse_request(cache_hint=CacheHint()) == parse_request(cache_hint=None)


def test_a_cached_content_resource_still_reaches_the_config() -> None:
    request = completion_request(
        cache_hint=CacheHint(), kwargs={"cached_content": "cachedContents/abc123"}
    )

    assert request["config"]["cached_content"] == "cachedContents/abc123"


def test_a_cached_content_resource_reaches_a_parse_config_too() -> None:
    request = parse_request(kwargs={"cached_content": "cachedContents/abc123"})

    assert request["config"]["cached_content"] == "cachedContents/abc123"


def test_instructions_leave_the_contents_for_the_config() -> None:
    request = completion_request()

    assert request["config"]["system_instruction"] == "You are terse."
    assert request["contents"] == [
        {"role": "user", "parts": [{"text": "Capital of France?"}]}
    ]


def test_no_instructions_means_no_system_instruction_field() -> None:
    request = completion_request(
        prompt=RenderedPrompt(messages=build_prompt().messages)
    )

    assert "system_instruction" not in request["config"]


def test_a_token_cap_is_sent_only_when_asked_for() -> None:
    assert completion_request(max_output_tokens=256)["config"]["max_output_tokens"] == (
        256
    )
    assert "max_output_tokens" not in completion_request()["config"]


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (ReasoningEffort.LOW, 1024),
        (ReasoningEffort.MEDIUM, 4096),
        (ReasoningEffort.HIGH, 16384),
        (ReasoningEffort.XHIGH, 24576),
        (ReasoningEffort.MAX, 32768),
    ],
)
def test_each_effort_becomes_the_token_budget_gemini_wants(
    effort: ReasoningEffort, budget: int
) -> None:
    config = completion_request(reasoning_effort=effort)["config"]

    assert config["thinking_config"] == {
        "thinking_budget": budget,
        "include_thoughts": True,
    }


def test_asking_for_no_reasoning_sends_no_thinking_config() -> None:
    config = completion_request(reasoning_effort=ReasoningEffort.NONE)["config"]

    assert "thinking_config" not in config


def test_an_explicit_thinking_config_is_left_alone() -> None:
    config = completion_request(
        reasoning_effort=ReasoningEffort.MAX,
        kwargs={"thinking_config": {"thinking_budget": 1}},
    )["config"]

    assert config["thinking_config"] == {"thinking_budget": 1}


def test_tools_are_gathered_into_a_single_declaration_list() -> None:
    config = completion_request(tools=[build_tool("alpha"), build_tool("beta")])[
        "config"
    ]

    assert len(config["tools"]) == 1
    assert [item["name"] for item in config["tools"][0]["function_declarations"]] == [
        "alpha",
        "beta",
    ]


def test_structured_output_is_a_response_schema_and_a_json_mime_type() -> None:
    config = parse_request()["config"]

    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"]["type"] == "object"
    assert "title" not in config["response_schema"]


def test_a_parse_declares_no_tools() -> None:
    assert "tools" not in parse_request()["config"]


def test_the_model_and_the_contents_sit_beside_the_config() -> None:
    request = completion_request()

    assert request["model"] == "gemini-test"
    assert request["contents"] is not None
    assert set(request) == {"model", "contents", "config"}


def test_extra_kwargs_reach_the_config_unaltered() -> None:
    config = completion_request(kwargs={"top_k": 5, "seed": 11})["config"]

    assert (config["top_k"], config["seed"]) == (5, 11)


def test_nothing_sets_a_temperature_unless_the_caller_does() -> None:
    assert (
        "temperature"
        not in completion_request(
            reasoning_effort=ReasoningEffort.HIGH, max_output_tokens=256
        )["config"]
    )
    assert completion_request(kwargs={"temperature": 0.3})["config"]["temperature"] == (
        0.3
    )


def test_a_model_name_is_required() -> None:
    with pytest.raises(LLMConfigError, match="model_name is required"):
        GeminiService(ModelConfigurations(api_host="http://x", api_key="k"))


def test_a_huge_instruction_is_sent_whole() -> None:
    instruction = "Follow the policy. " * 5000
    request = completion_request(
        prompt=RenderedPrompt(
            instructions=[
                RenderedPromptInstruction(role=MessageRole.SYSTEM, content=instruction)
            ],
            messages=build_prompt().messages,
        )
    )

    assert request["config"]["system_instruction"] == instruction
