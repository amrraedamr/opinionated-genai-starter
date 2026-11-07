"""Proves each adapter still fits the API it was written against.

One gateway and one key serve all three providers, so these differ only in
model name and in which service is pointed at it. They skip themselves when
the gateway is unreachable, which is the usual case off the VPN.

`max_output_tokens` is generous because a reasoning model spends its budget
before it answers, and a cap too low returns an empty reply rather than a
short one. Note that `effort=max` is rejected on Chat Completions and accepted
on Responses: that is the provider's limitation, not this seam's.
"""

from typing import Any

import pytest

from base.models.definitions.model import ProviderType, ReasoningEffort
from base.models.view_models import BaseModel
from base.models.view_models.model import CacheHint, ModelConfigurations
from base.services.configuration.mixins import (
    AnthropicSettingsMixin,
    GeminiSettingsMixin,
    LLMSettingsMixin,
    OpenAILLMSettingsMixin,
)
from base.services.llm.anthropic import AnthropicService
from base.services.llm.gemini import GeminiService
from base.services.llm.openai import OpenAIService
from tests.conftest import credential
from tests.fakes import build_prompt, build_tool

BUDGET: int = 2048
"""A reasoning model spends its cap before answering, so this is not small."""

OPENAI_MODEL = "azure.gpt-5.6-terra"
ANTHROPIC_MODEL = "bedrock.anthropic.claude-haiku-5-5"
GEMINI_MODEL = "vertex_ai.gemini-3.8-flash"


class Capital(BaseModel):
    """A small structured answer, cheap to ask for."""

    city: str
    country: str


def gateway(model: str, provider: ProviderType, **extra: Any) -> ModelConfigurations:
    """The shared gateway, pointed at one model."""
    return ModelConfigurations(
        api_host=credential("OPENAI_LLM_ENDPOINT"),
        api_key=credential("OPENAI_LLM_API_KEY"),
        model_name=model,
        api_type=provider,
        api_retries=1,
        **extra,
    )


@pytest.mark.live
async def test_the_responses_surface_answers(make_settings: Any) -> None:
    make_settings(LLMSettingsMixin, OpenAILLMSettingsMixin)
    service = OpenAIService(
        gateway(OPENAI_MODEL, ProviderType.OPENAI, supports_responses_api=True),
        _key="live-responses",
    )

    result = await service.acomplete(
        prompt=build_prompt(question="Capital of France? One word."),
        max_output_tokens=BUDGET,
    )

    assert result.content
    assert result.usage is not None
    assert result.usage.input_tokens > 0


@pytest.mark.live
async def test_the_chat_completions_surface_answers(make_settings: Any) -> None:
    make_settings(LLMSettingsMixin, OpenAILLMSettingsMixin)
    service = OpenAIService(
        gateway(OPENAI_MODEL, ProviderType.OPENAI, supports_responses_api=False),
        _key="live-completions",
    )

    result = await service.acomplete(
        prompt=build_prompt(question="Capital of France? One word."),
        max_output_tokens=BUDGET,
        reasoning_effort=ReasoningEffort.LOW,
    )

    assert result.content


@pytest.mark.live
async def test_the_responses_surface_accepts_the_maximum_effort(
    make_settings: Any,
) -> None:
    """Accepted here and rejected on Chat Completions, by the provider's rules."""
    make_settings(LLMSettingsMixin, OpenAILLMSettingsMixin)
    service = OpenAIService(
        gateway(OPENAI_MODEL, ProviderType.OPENAI, supports_responses_api=True),
        _key="live-responses-max",
    )

    result = await service.acomplete(
        prompt=build_prompt(question="Capital of France? One word."),
        max_output_tokens=BUDGET,
        reasoning_effort=ReasoningEffort.MAX,
    )

    assert result.content


@pytest.mark.live
async def test_a_cache_hint_is_accepted_by_the_gateway(make_settings: Any) -> None:
    make_settings(LLMSettingsMixin, OpenAILLMSettingsMixin)
    service = OpenAIService(
        gateway(OPENAI_MODEL, ProviderType.OPENAI, supports_responses_api=True),
        _key="live-cache-hint",
    )

    result = await service.acomplete(
        prompt=build_prompt(question="Capital of France? One word."),
        max_output_tokens=BUDGET,
        cache_hint=CacheHint(),
    )

    assert result.content


@pytest.mark.live
async def test_a_parsed_call_returns_the_schema_it_asked_for(
    make_settings: Any,
) -> None:
    make_settings(LLMSettingsMixin, OpenAILLMSettingsMixin)
    service = OpenAIService(
        gateway(OPENAI_MODEL, ProviderType.OPENAI, supports_responses_api=True),
        _key="live-parse",
    )

    result = await service.aparse(
        Capital,
        prompt=build_prompt(question="The capital of France, and its country."),
        max_output_tokens=BUDGET,
    )

    assert result.parsed.city
    assert result.parsed.country


@pytest.mark.live
async def test_a_tool_is_offered_and_can_be_asked_for(make_settings: Any) -> None:
    make_settings(LLMSettingsMixin, OpenAILLMSettingsMixin)
    service = OpenAIService(
        gateway(OPENAI_MODEL, ProviderType.OPENAI, supports_responses_api=True),
        _key="live-tools",
    )

    result = await service.acomplete(
        prompt=build_prompt(
            instruction="Use the weather tool when asked about weather.",
            question="What is the weather in Paris?",
        ),
        tools=[build_tool()],
        max_output_tokens=BUDGET,
    )

    assert result.tool_calls or result.content


@pytest.mark.live
async def test_anthropics_native_messages_api_answers(make_settings: Any) -> None:
    make_settings(LLMSettingsMixin, AnthropicSettingsMixin)
    service = AnthropicService(
        gateway(ANTHROPIC_MODEL, ProviderType.ANTHROPIC), _key="live-anthropic"
    )

    result = await service.acomplete(
        prompt=build_prompt(question="Capital of France? One word."),
        max_output_tokens=BUDGET,
    )

    assert result.content
    assert result.usage is not None
    assert result.usage.input_tokens > 0


@pytest.mark.live
async def test_anthropic_returns_the_schema_it_was_given(make_settings: Any) -> None:
    make_settings(LLMSettingsMixin, AnthropicSettingsMixin)
    service = AnthropicService(
        gateway(ANTHROPIC_MODEL, ProviderType.ANTHROPIC), _key="live-anthropic-parse"
    )

    result = await service.aparse(
        Capital,
        prompt=build_prompt(question="The capital of France, and its country."),
        max_output_tokens=BUDGET,
    )

    assert result.parsed.city


@pytest.mark.live
async def test_gemini_answers(make_settings: Any) -> None:
    make_settings(LLMSettingsMixin, GeminiSettingsMixin)
    service = GeminiService(
        gateway(GEMINI_MODEL, ProviderType.GEMINI), _key="live-gemini"
    )

    result = await service.acomplete(
        prompt=build_prompt(question="Capital of France? One word."),
        max_output_tokens=BUDGET,
    )

    assert result.content
    assert result.usage is not None
    assert result.usage.input_tokens > 0


@pytest.mark.live
async def test_gemini_returns_the_schema_it_was_given(make_settings: Any) -> None:
    make_settings(LLMSettingsMixin, GeminiSettingsMixin)
    service = GeminiService(
        gateway(GEMINI_MODEL, ProviderType.GEMINI), _key="live-gemini-parse"
    )

    result = await service.aparse(
        Capital,
        prompt=build_prompt(question="The capital of France, and its country."),
        max_output_tokens=BUDGET,
    )

    assert result.parsed.city
