"""Pins the request OpenAI would have been sent, for both of its surfaces.

`OpenAIService` speaks either the Responses API or Chat Completions, and the
difference shows up as spelling: the same neutral knob becomes
`max_output_tokens` on one and `max_completion_tokens` on the other. Prompt
caching is the other half — OpenAI caches a repeated prefix regardless, so the
hint buys routing, and the routing key has to name the head without naming the
conversation that grows beneath it.
"""

from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.definitions.model import CachePersistence, ReasoningEffort
from base.models.view_models import BaseModel
from base.models.view_models.model import CacheHint, ModelConfigurations
from base.models.view_models.prompt import (
    RenderedPrompt,
    RenderedPromptInstruction,
    RenderedPromptMessage,
)
from base.services.llm.openai import OpenAIService
from tests.fakes import build_prompt, build_tool, stored_message


class Answer(BaseModel):
    """A one-field target, so a schema change is a single edit."""

    text: str


class OtherAnswer(BaseModel):
    """A differently shaped target."""

    number: int


def bare_service(responses_api: bool = True) -> OpenAIService:
    """A service with only the attributes the shaping methods read."""
    service = object.__new__(OpenAIService)
    service._supports_responses_api = responses_api
    service._model = "gpt-test"
    service._auth = None
    return service


@pytest.mark.parametrize(
    ("responses_api", "token_key", "effort"),
    [
        (True, "max_output_tokens", {"reasoning": {"effort": ReasoningEffort.HIGH}}),
        (False, "max_completion_tokens", {"reasoning_effort": ReasoningEffort.HIGH}),
    ],
)
def test_each_surface_spells_the_neutral_knobs_its_own_way(
    responses_api: bool, token_key: str, effort: dict[str, Any]
) -> None:
    kwargs: dict[str, Any] = {}

    bare_service(responses_api)._apply_generation_knobs(
        kwargs, 512, ReasoningEffort.HIGH
    )

    assert kwargs == {token_key: 512, **effort}


def test_the_responses_surface_nests_the_effort_and_completions_keeps_it_flat() -> None:
    nested: dict[str, Any] = {}
    flat: dict[str, Any] = {}

    bare_service(True)._apply_generation_knobs(nested, None, ReasoningEffort.MEDIUM)
    bare_service(False)._apply_generation_knobs(flat, None, ReasoningEffort.MEDIUM)

    assert nested["reasoning"] == {"effort": ReasoningEffort.MEDIUM}
    assert "reasoning_effort" not in nested
    assert flat["reasoning_effort"] == ReasoningEffort.MEDIUM
    assert "reasoning" not in flat


def test_omitted_knobs_add_nothing() -> None:
    kwargs: dict[str, Any] = {}

    bare_service()._apply_generation_knobs(kwargs, None, None)

    assert kwargs == {}


def test_an_explicit_knob_survives_the_neutral_one() -> None:
    kwargs: dict[str, Any] = {
        "max_output_tokens": 1,
        "reasoning": {"effort": "low", "summary": "auto"},
    }

    bare_service()._apply_generation_knobs(kwargs, 9999, ReasoningEffort.MAX)

    assert kwargs["max_output_tokens"] == 1
    assert kwargs["reasoning"] == {"effort": "low", "summary": "auto"}


@pytest.mark.parametrize("effort", list(ReasoningEffort))
def test_every_effort_including_none_is_passed_through_verbatim(
    effort: ReasoningEffort,
) -> None:
    kwargs: dict[str, Any] = {}

    bare_service()._apply_generation_knobs(kwargs, None, effort)

    assert kwargs == {"reasoning": {"effort": effort}}


@pytest.mark.parametrize("responses_api", [True, False])
def test_nothing_sets_a_temperature_unless_the_caller_does(
    responses_api: bool,
) -> None:
    kwargs: dict[str, Any] = {}
    service = bare_service(responses_api)

    service._apply_generation_knobs(kwargs, 512, ReasoningEffort.HIGH)
    service._apply_cache_hint(kwargs, CacheHint(), build_prompt())

    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


@pytest.mark.parametrize("responses_api", [True, False])
def test_a_completion_sends_no_temperature_the_caller_did_not_ask_for(
    responses_api: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: dict[str, Any] = {}
    service = bare_service(responses_api)

    def capture(_input: Any, **kwargs: Any) -> Any:
        sent.update(kwargs)
        raise _Captured

    monkeypatch.setattr(
        service, "responses" if responses_api else "completions", capture
    )
    with pytest.raises(_Captured):
        service._complete(
            build_prompt(), None, None, None, 256, ReasoningEffort.LOW, None, None
        )

    assert "temperature" not in sent


def test_a_caller_asking_for_a_temperature_still_gets_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}
    service = bare_service(True)

    def capture(_input: Any, **kwargs: Any) -> Any:
        sent.update(kwargs)
        raise _Captured

    monkeypatch.setattr(service, "responses", capture)
    with pytest.raises(_Captured):
        service._complete(
            build_prompt(), None, None, None, None, None, None, None, temperature=0.3
        )

    assert sent["temperature"] == 0.3


class _Captured(Exception):
    """Stops the call once the request is in hand."""


def test_a_cache_hint_asks_for_routing_and_a_retention_tier() -> None:
    request: dict[str, Any] = {}

    bare_service()._apply_cache_hint(request, CacheHint(), build_prompt())

    assert set(request) == {"prompt_cache_key", "prompt_cache_retention"}
    assert request["prompt_cache_retention"] == "in_memory"
    assert len(request["prompt_cache_key"]) == 32


def test_no_cache_hint_asks_for_nothing() -> None:
    request: dict[str, Any] = {}

    bare_service()._apply_cache_hint(
        request, None, build_prompt(), tools=[build_tool()]
    )

    assert request == {}


@pytest.mark.parametrize(
    ("persistence", "retention"),
    [(CachePersistence.SHORT, "in_memory"), (CachePersistence.LONG, "24h")],
)
def test_each_persistence_level_becomes_its_storage_tier(
    persistence: CachePersistence, retention: str
) -> None:
    request: dict[str, Any] = {}

    bare_service()._apply_cache_hint(
        request, CacheHint(persistence=persistence), build_prompt()
    )

    assert request["prompt_cache_retention"] == retention


def test_an_explicit_hint_key_wins_over_the_fingerprint() -> None:
    request: dict[str, Any] = {}

    bare_service()._apply_cache_hint(
        request, CacheHint(key="conversation-7"), build_prompt()
    )

    assert request["prompt_cache_key"] == "conversation-7"


def test_an_explicit_kwarg_wins_over_both() -> None:
    request: dict[str, Any] = {
        "prompt_cache_key": "caller-chose",
        "prompt_cache_retention": "24h",
    }

    bare_service()._apply_cache_hint(
        request, CacheHint(key="hint-chose"), build_prompt()
    )

    assert request["prompt_cache_key"] == "caller-chose"
    assert request["prompt_cache_retention"] == "24h"


def test_the_same_head_fingerprints_the_same_on_either_surface() -> None:
    prompt = build_prompt()
    tools = [build_tool()]

    assert OpenAIService._head_fingerprint(
        prompt, tools
    ) == OpenAIService._head_fingerprint(build_prompt(), [build_tool()])


def test_different_instructions_fingerprint_differently() -> None:
    assert OpenAIService._head_fingerprint(
        build_prompt("You are terse."), None
    ) != OpenAIService._head_fingerprint(build_prompt("You are verbose."), None)


def test_tool_order_is_part_of_the_fingerprint() -> None:
    first, second = build_tool("alpha"), build_tool("beta")

    assert OpenAIService._head_fingerprint(
        None, [first, second]
    ) != OpenAIService._head_fingerprint(None, [second, first])


def test_an_output_schema_is_part_of_the_fingerprint() -> None:
    prompt = build_prompt()

    assert OpenAIService._head_fingerprint(
        prompt, None, Answer
    ) != OpenAIService._head_fingerprint(prompt, None, OtherAnswer)
    assert OpenAIService._head_fingerprint(
        prompt, None, Answer
    ) != OpenAIService._head_fingerprint(prompt, None, None)


def test_a_growing_conversation_does_not_change_the_fingerprint() -> None:
    prompt = build_prompt()
    short = [stored_message(MessageRole.USER, "one", 1)]
    long = short + [
        stored_message(MessageRole.ASSISTANT, "two", 2),
        stored_message(MessageRole.USER, "three", 3),
        stored_message(MessageRole.ASSISTANT, "four", 4),
    ]

    assert OpenAIService._head_fingerprint(
        prompt, None, None, short
    ) == OpenAIService._head_fingerprint(prompt, None, None, long)


@pytest.mark.parametrize("role", [MessageRole.SYSTEM, MessageRole.DEVELOPER])
def test_an_instruction_arriving_as_a_stored_message_counts(
    role: MessageRole,
) -> None:
    """Regression: instructions off the prompt once vanished from the key."""
    plain = OpenAIService._head_fingerprint(None, None, None, [stored_message()])
    instructed = OpenAIService._head_fingerprint(
        None,
        None,
        None,
        [stored_message(), stored_message(role, "You are terse.", 2)],
    )

    assert plain != instructed


def test_a_stored_instruction_fingerprints_as_the_same_head_as_a_prompt_one() -> None:
    from_prompt = OpenAIService._head_fingerprint(build_prompt("You are terse."), None)
    from_history = OpenAIService._head_fingerprint(
        None, None, None, [stored_message(MessageRole.SYSTEM, "You are terse.")]
    )

    assert from_prompt == from_history


def test_two_conversations_with_different_stored_instructions_route_apart() -> None:
    terse = OpenAIService._head_fingerprint(
        None, None, None, [stored_message(MessageRole.SYSTEM, "Be terse.")]
    )
    verbose = OpenAIService._head_fingerprint(
        None, None, None, [stored_message(MessageRole.SYSTEM, "Be verbose.")]
    )

    assert terse != verbose


def test_an_empty_head_still_fingerprints() -> None:
    assert len(OpenAIService._head_fingerprint(None, None, None, None)) == 32


def test_the_cache_key_tracks_the_stored_instructions_the_call_carried() -> None:
    service = bare_service()
    without: dict[str, Any] = {}
    with_instruction: dict[str, Any] = {}

    service._apply_cache_hint(without, CacheHint(), None, messages=[stored_message()])
    service._apply_cache_hint(
        with_instruction,
        CacheHint(),
        None,
        messages=[stored_message(MessageRole.SYSTEM, "Be terse.")],
    )

    assert without["prompt_cache_key"] != with_instruction["prompt_cache_key"]


def test_a_parse_call_keys_on_its_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}
    service = bare_service(True)

    def capture(_input: Any, _schema: Any, **kwargs: Any) -> Any:
        sent.update(kwargs)
        raise _Captured

    monkeypatch.setattr(service, "parse_responses", capture)
    with pytest.raises(_Captured):
        service._parse(Answer, build_prompt(), None, None, None, None, CacheHint())

    assert sent["prompt_cache_key"] == OpenAIService._head_fingerprint(
        build_prompt(), None, Answer, None
    )


def test_a_completion_keys_on_its_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}
    service = bare_service(True)
    tools = [build_tool()]

    def capture(_input: Any, **kwargs: Any) -> Any:
        sent.update(kwargs)
        raise _Captured

    monkeypatch.setattr(service, "responses", capture)
    with pytest.raises(_Captured):
        service._complete(
            build_prompt(), None, tools, None, None, None, None, CacheHint()
        )

    assert sent["prompt_cache_key"] == OpenAIService._head_fingerprint(
        build_prompt(), tools, None, None
    )


@pytest.mark.parametrize(
    ("responses_api", "shape"),
    [
        (True, {"type": "function", "name": "get_weather"}),
        (False, {"type": "function"}),
    ],
)
def test_each_surface_sends_tools_in_its_own_shape(
    responses_api: bool, shape: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: dict[str, Any] = {}
    service = bare_service(responses_api)

    def capture(_input: Any, **kwargs: Any) -> Any:
        sent.update(kwargs)
        raise _Captured

    monkeypatch.setattr(
        service, "responses" if responses_api else "completions", capture
    )
    with pytest.raises(_Captured):
        service._complete(
            build_prompt(), None, [build_tool()], None, None, None, None, None
        )

    tool = sent["tools"][0]
    assert {key: tool[key] for key in shape} == shape
    assert ("function" in tool) is not responses_api


def test_a_model_name_is_required() -> None:
    from base.models.view_models.exceptions.llm import LLMConfigError

    with pytest.raises(LLMConfigError, match="model_name is required"):
        OpenAIService(ModelConfigurations(api_host="http://x", api_key="k"))


def test_an_async_credential_refuses_the_sync_surface() -> None:
    from base.models.view_models.exceptions.llm import LLMConfigError

    service = bare_service()
    service._auth = object()

    with pytest.raises(LLMConfigError, match="no synchronous surface"):
        service._complete(build_prompt(), None, None, None, None, None, None, None)


def test_a_prompt_carrying_only_instructions_still_shapes_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = bare_service(True)
    prompt = RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content="Be terse.")
        ]
    )
    seen: list[Any] = []

    def capture(input_: Any, **_kwargs: Any) -> Any:
        seen.append(input_)
        raise _Captured

    monkeypatch.setattr(service, "responses", capture)
    with pytest.raises(_Captured):
        service._complete(prompt, None, None, None, None, None, None, None)

    assert seen[0] == [{"role": "system", "content": "Be terse."}]


def test_a_prompt_and_messages_together_arrive_prompt_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = bare_service(True)
    prompt = RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content="Be terse.")
        ],
        messages=[RenderedPromptMessage(role=MessageRole.USER, content="from prompt")],
    )
    seen: list[Any] = []

    def capture(input_: Any, **_kwargs: Any) -> Any:
        seen.append(input_)
        raise _Captured

    monkeypatch.setattr(service, "responses", capture)
    with pytest.raises(_Captured):
        service._complete(
            prompt,
            [stored_message(MessageRole.USER, "from history")],
            None,
            None,
            None,
            None,
            None,
            None,
        )

    assert [item["content"] for item in seen[0]] == [
        "Be terse.",
        "from prompt",
        "from history",
    ]
