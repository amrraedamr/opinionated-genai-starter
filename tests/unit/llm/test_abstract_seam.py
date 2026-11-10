"""Pins the contract `AbstractLLMService` imposes on every adapter.

The public four methods are `@final`: they validate, then hand everything to
the `_`-prefixed implementation positionally. An adapter that reorders or
renames a parameter would still type-check, so the order is pinned here, and
the three concrete services are checked against the abstract signature rather
than against each other.
"""

import inspect
from collections.abc import Sequence
from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.definitions.model import ReasoningEffort
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredMessage
from base.models.view_models.llm_result import LLMResult, ParsedResult
from base.models.view_models.model import CacheHint
from base.models.view_models.prompt import RenderedPrompt
from base.services.llm import AbstractLLMService
from base.services.llm.anthropic import AnthropicService
from base.services.llm.gemini import GeminiService
from base.services.llm.openai import OpenAIService
from base.services.tool_calling import Tool
from base.tools.stream import StreamIterator
from tests.fakes import build_prompt, build_tool, stored_message

SERVICES: list[type[AbstractLLMService]] = [
    OpenAIService,
    AnthropicService,
    GeminiService,
]

FORWARDED: list[str] = ["_complete", "_acomplete", "_parse", "_aparse"]


class Answer(BaseModel):
    """Target schema for the parse surface."""

    text: str


class RecordingService(AbstractLLMService):
    """A concrete service that keeps the arguments the seam handed it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    @property
    def model(self) -> str:
        return "recorder"

    def get_service_name(self) -> str:
        return "recorder"

    def _apply_cache_hint(
        self,
        request: dict[str, Any],
        cache_hint: CacheHint | None,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None = None,
        tools: Sequence[Tool[BaseModel]] | None = None,
        schema: type[BaseModel] | None = None,
    ) -> None:
        self.calls.append(("_apply_cache_hint", (request, cache_hint, prompt), {}))

    def _complete(self, *args: Any, **kwargs: Any) -> LLMResult:
        self.calls.append(("_complete", args, kwargs))
        return LLMResult(content="ok")

    async def _acomplete(self, *args: Any, **kwargs: Any) -> LLMResult:
        self.calls.append(("_acomplete", args, kwargs))
        return LLMResult(content="ok")

    def _parse(self, *args: Any, **kwargs: Any) -> ParsedResult[Any]:
        self.calls.append(("_parse", args, kwargs))
        return ParsedResult[Answer](parsed=Answer(text="ok"))

    async def _aparse(self, *args: Any, **kwargs: Any) -> ParsedResult[Any]:
        self.calls.append(("_aparse", args, kwargs))
        return ParsedResult[Answer](parsed=Answer(text="ok"))

    @property
    def last(self) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        assert self.calls, "nothing was forwarded"
        return self.calls[-1]


def _completion_arguments() -> dict[str, Any]:
    """Every argument `complete` accepts, each a distinguishable object."""
    return {
        "prompt": build_prompt(),
        "messages": [stored_message()],
        "tools": [build_tool()],
        "stream_iterator": StreamIterator(),
        "max_output_tokens": 256,
        "reasoning_effort": ReasoningEffort.HIGH,
        "label": "a-label",
        "cache_hint": CacheHint(),
    }


def test_complete_forwards_every_argument_positionally_and_in_order() -> None:
    service = RecordingService()
    given = _completion_arguments()

    service.complete(**given, passthrough="kept")

    name, args, kwargs = service.last
    assert name == "_complete"
    assert args == tuple(given.values())
    assert kwargs == {"passthrough": "kept"}


async def test_acomplete_forwards_every_argument_positionally_and_in_order() -> None:
    service = RecordingService()
    given = _completion_arguments()

    await service.acomplete(**given, passthrough="kept")

    name, args, kwargs = service.last
    assert name == "_acomplete"
    assert args == tuple(given.values())
    assert kwargs == {"passthrough": "kept"}


def test_parse_forwards_the_schema_first_then_every_argument_in_order() -> None:
    service = RecordingService()
    given: dict[str, Any] = {
        "prompt": build_prompt(),
        "messages": [stored_message()],
        "max_output_tokens": 256,
        "reasoning_effort": ReasoningEffort.LOW,
        "label": "a-label",
        "cache_hint": CacheHint(),
    }

    service.parse(Answer, **given, passthrough="kept")

    name, args, kwargs = service.last
    assert name == "_parse"
    assert args == (Answer, *given.values())
    assert kwargs == {"passthrough": "kept"}


async def test_aparse_forwards_the_schema_first_then_every_argument_in_order() -> None:
    service = RecordingService()
    given: dict[str, Any] = {
        "prompt": build_prompt(),
        "messages": [stored_message()],
        "max_output_tokens": 256,
        "reasoning_effort": ReasoningEffort.LOW,
        "label": "a-label",
        "cache_hint": CacheHint(),
    }

    await service.aparse(Answer, **given, passthrough="kept")

    name, args, kwargs = service.last
    assert name == "_aparse"
    assert args == (Answer, *given.values())
    assert kwargs == {"passthrough": "kept"}


def test_a_call_that_omits_both_inputs_never_reaches_the_adapter() -> None:
    service = RecordingService()

    with pytest.raises(ValueError, match="complete needs a prompt, messages, or both"):
        service.complete()

    assert service.calls == []


async def test_an_async_call_that_omits_both_inputs_names_the_method_it_came_from() -> (
    None
):
    service = RecordingService()

    with pytest.raises(ValueError, match="^acomplete needs"):
        await service.acomplete()
    with pytest.raises(ValueError, match="^aparse needs"):
        await service.aparse(Answer)


def test_an_empty_message_list_counts_as_no_input() -> None:
    service = RecordingService()

    with pytest.raises(ValueError, match="parse needs"):
        service.parse(Answer, messages=[])


def test_messages_alone_are_enough_to_proceed() -> None:
    service = RecordingService()

    service.complete(messages=[stored_message()])

    assert service.last[0] == "_complete"


@pytest.mark.parametrize("service", SERVICES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("method", FORWARDED)
def test_every_service_matches_the_abstract_signature(
    service: type[AbstractLLMService], method: str
) -> None:
    expected = inspect.signature(getattr(AbstractLLMService, method))
    actual = inspect.signature(getattr(service, method))

    assert [p.name for p in actual.parameters.values()] == [
        p.name for p in expected.parameters.values()
    ]
    assert [p.kind for p in actual.parameters.values()] == [
        p.kind for p in expected.parameters.values()
    ]


@pytest.mark.parametrize("service", SERVICES, ids=lambda cls: cls.__name__)
def test_every_service_matches_the_abstract_cache_hint_signature(
    service: type[AbstractLLMService],
) -> None:
    expected = inspect.signature(AbstractLLMService._apply_cache_hint)
    actual = inspect.signature(service._apply_cache_hint)

    assert actual.parameters.keys() == expected.parameters.keys()
    assert [p.default for p in actual.parameters.values()] == [
        p.default for p in expected.parameters.values()
    ]


def test_applying_a_cache_hint_is_abstract() -> None:
    assert "_apply_cache_hint" in AbstractLLMService.__abstractmethods__


def test_a_subclass_omitting_the_cache_hint_cannot_be_instantiated() -> None:
    class Forgetful(AbstractLLMService):
        """Everything but `_apply_cache_hint`, which is the point."""

        @property
        def model(self) -> str:
            return "forgetful"

        def get_service_name(self) -> str:
            return "forgetful"

        def _complete(self, *args: Any, **kwargs: Any) -> LLMResult:
            return LLMResult()

        async def _acomplete(self, *args: Any, **kwargs: Any) -> LLMResult:
            return LLMResult()

        def _parse(self, *args: Any, **kwargs: Any) -> ParsedResult[Any]:
            raise NotImplementedError

        async def _aparse(self, *args: Any, **kwargs: Any) -> ParsedResult[Any]:
            raise NotImplementedError

    assert Forgetful.__abstractmethods__ == frozenset({"_apply_cache_hint"})
    with pytest.raises(TypeError, match="_apply_cache_hint"):
        Forgetful()


@pytest.mark.parametrize("service", SERVICES, ids=lambda cls: cls.__name__)
def test_no_service_leaves_an_abstract_method_unimplemented(
    service: type[AbstractLLMService],
) -> None:
    assert service.__abstractmethods__ == frozenset()


@pytest.mark.parametrize("method", ["complete", "acomplete", "parse", "aparse"])
def test_the_public_surface_is_final(method: str) -> None:
    assert getattr(AbstractLLMService, method).__final__ is True


@pytest.mark.parametrize("service", SERVICES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("method", ["complete", "acomplete", "parse", "aparse"])
def test_no_service_overrides_the_public_surface(
    service: type[AbstractLLMService], method: str
) -> None:
    assert getattr(service, method) is getattr(AbstractLLMService, method)


def test_the_seam_names_the_three_service_names_apart() -> None:
    names = {service.get_service_name(object.__new__(service)) for service in SERVICES}
    assert names == {"openai", "anthropic", "gemini"}


def test_an_instruction_only_prompt_is_accepted_without_messages() -> None:
    service = RecordingService()
    prompt = RenderedPrompt(
        instructions=build_prompt().instructions,
        messages=[],
    )

    service.complete(prompt=prompt)

    assert service.last[1][0] is prompt


def test_a_developer_role_message_is_input_like_any_other() -> None:
    service = RecordingService()
    message = stored_message(MessageRole.DEVELOPER, "Answer in French.")

    service.complete(messages=[message])

    assert service.last[1][1] == [message]
