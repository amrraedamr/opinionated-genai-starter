"""What only a real provider can answer.

The unit suite proves what each adapter *builds*. These prove what the
providers do with it, which is a different question and the one that has
caught every provider bug in this repository so far: a parameter the SDK
types accept and the service rejects, a URL rewritten into one that 401s, a
cache that reports nothing because the prefix moved.

Every test here costs money, so prompts are a few tokens and the models are
the cheap tier. Reasoning models spend their budget before answering, so
`max_output_tokens` is generous even for a one-word reply.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from base.models.definitions.message import MessageRole
from base.models.definitions.model import (
    CachePersistence,
    FinishReason,
    ProviderType,
    ReasoningEffort,
)
from base.models.view_models import BaseModel
from base.models.view_models.model import CacheHint, ModelConfigurations
from base.models.view_models.prompt import (
    RenderedPrompt,
    RenderedPromptInstruction,
    RenderedPromptMessage,
)
from base.services.configuration.mixins import (
    AnthropicSettingsMixin,
    GeminiSettingsMixin,
    LLMSettingsMixin,
    OpenAILLMSettingsMixin,
)
from base.services.llm.anthropic import AnthropicService
from base.services.llm.gemini import GeminiService
from base.services.llm.openai import OpenAIService
from base.services.tool_calling import Tool
from tests.conftest import credential

pytestmark = pytest.mark.live

OPENAI_MODEL = "azure.gpt-5.6-terra"
ANTHROPIC_MODEL = "bedrock.anthropic.claude-haiku-5-5"
GEMINI_MODEL = "vertex_ai.gemini-3.8-flash"

BUDGET = 2048
"""Enough that a reasoning model still has room to answer after thinking."""


class Capital(BaseModel):
    city: str
    country: str


class WeatherArgs(BaseModel):
    city: str


async def _weather(args: WeatherArgs) -> str:
    return f"It is 21 degrees and clear in {args.city}."


def weather_tool() -> Tool[Any]:
    return Tool(
        name="get_weather",
        description="Current weather for a city.",
        parameters=WeatherArgs,
        handler=_weather,
    )


def ask(question: str, instruction: str = "Answer in one word.") -> RenderedPrompt:
    return RenderedPrompt(
        instructions=[
            RenderedPromptInstruction(role=MessageRole.SYSTEM, content=instruction)
        ],
        messages=[RenderedPromptMessage(role=MessageRole.USER, content=question)],
    )


def configure_all(make_settings: Any) -> None:
    """One gateway and one key serve every provider here."""
    make_settings(
        LLMSettingsMixin,
        OpenAILLMSettingsMixin,
        AnthropicSettingsMixin,
        GeminiSettingsMixin,
    )


def config_for(provider: ProviderType, model: str) -> ModelConfigurations:
    return ModelConfigurations(
        api_host=credential("OPENAI_LLM_ENDPOINT"),
        api_key=credential("OPENAI_LLM_API_KEY"),
        model_name=model,
        api_type=provider,
        api_retries=1,
        wait_time=0.5,
    )


def openai_service(*, responses: bool, label: str) -> OpenAIService:
    config = config_for(ProviderType.OPENAI, f"{OPENAI_MODEL}#{label}")
    config.model_name = OPENAI_MODEL
    config.supports_responses_api = responses
    return OpenAIService(config)


def every_provider() -> list[tuple[str, Any]]:
    """The three adapters, each configured against the gateway."""
    return [
        ("openai-completions", openai_service(responses=False, label="x1")),
        ("openai-responses", openai_service(responses=True, label="x2")),
        (
            "anthropic",
            AnthropicService(config_for(ProviderType.ANTHROPIC, ANTHROPIC_MODEL)),
        ),
        ("gemini", GeminiService(config_for(ProviderType.GEMINI, GEMINI_MODEL))),
    ]


async def test_every_provider_answers_the_same_neutral_call(make_settings: Any) -> None:
    """The seam's whole claim: one call shape, whoever is behind it."""
    configure_all(make_settings)
    answers: dict[str, str] = {}
    for name, service in every_provider():
        result = await service.acomplete(
            prompt=ask("What is the capital of France?"), max_output_tokens=BUDGET
        )
        assert result.content, f"{name} returned nothing"
        answers[name] = result.content.strip().lower()

    assert len(answers) == 4
    assert all("paris" in answer for answer in answers.values()), answers


async def test_usage_is_reported_and_adds_up(make_settings: Any) -> None:
    """Both cache counts are shares of the input, and reasoning of the output."""
    configure_all(make_settings)
    for name, service in every_provider():
        result = await service.acomplete(
            prompt=ask("Name one colour."), max_output_tokens=BUDGET
        )
        usage = result.usage
        assert usage is not None, f"{name} reported no usage"
        assert usage.input_tokens > 0, f"{name} counted no input"
        assert usage.output_tokens > 0, f"{name} counted no output"
        assert usage.cached_input_tokens <= usage.input_tokens, name
        assert usage.cache_write_tokens <= usage.input_tokens, name
        assert usage.reasoning_tokens <= usage.output_tokens, name
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens, name


async def test_a_finish_reason_is_normalised_for_every_provider(
    make_settings: Any,
) -> None:
    """A caller branches on this, so it must never arrive provider-shaped."""
    configure_all(make_settings)
    for name, service in every_provider():
        result = await service.acomplete(
            prompt=ask("Name one colour."), max_output_tokens=BUDGET
        )
        assert isinstance(result.finish_reason, FinishReason), (
            f"{name} returned {result.finish_reason!r}"
        )


async def test_a_truncated_reply_says_it_was_truncated(make_settings: Any) -> None:
    """LENGTH is the one finish reason a caller must act on."""
    configure_all(make_settings)
    service = AnthropicService(config_for(ProviderType.ANTHROPIC, ANTHROPIC_MODEL))
    result = await service.acomplete(
        prompt=ask("Count slowly from one to two hundred.", instruction="Be verbose."),
        max_output_tokens=16,
    )
    assert result.finish_reason is FinishReason.LENGTH


async def test_reasoning_effort_buys_reasoning(make_settings: Any) -> None:
    """A higher effort must reach the provider, not just the request body."""
    configure_all(make_settings)
    service = openai_service(responses=True, label="effort")
    low = await service.acomplete(
        prompt=ask("What is 17 times 23?"),
        max_output_tokens=4096,
        reasoning_effort=ReasoningEffort.LOW,
    )
    assert low.content and "391" in low.content


@pytest.mark.parametrize(
    "effort",
    [ReasoningEffort.NONE, ReasoningEffort.LOW, ReasoningEffort.HIGH],
)
async def test_every_effort_level_is_accepted_by_anthropic(
    make_settings: Any, effort: ReasoningEffort
) -> None:
    """`output_config.effort` replaced a token budget its models now reject."""
    configure_all(make_settings)
    service = AnthropicService(config_for(ProviderType.ANTHROPIC, ANTHROPIC_MODEL))
    result = await service.acomplete(
        prompt=ask("Capital of Italy?"),
        max_output_tokens=BUDGET,
        reasoning_effort=effort,
    )
    assert result.content and "rome" in result.content.lower()


async def test_the_maximum_effort_reaches_the_responses_surface(
    make_settings: Any,
) -> None:
    """`max` is the level this repo added last; Chat Completions rejects it."""
    configure_all(make_settings)
    service = openai_service(responses=True, label="max")
    result = await service.acomplete(
        prompt=ask("Capital of Spain?"),
        max_output_tokens=4096,
        reasoning_effort=ReasoningEffort.MAX,
    )
    assert result.content and "madrid" in result.content.lower()


async def test_a_repeated_prefix_is_actually_cached(make_settings: Any) -> None:
    """The point of a cache hint, and the only way to know it worked.

    A hit needs a head long enough to clear each provider's minimum and a
    second call close behind the first, so both are sent here rather than
    assumed from one.
    """
    configure_all(make_settings)
    head = "You are a careful assistant. " * 240
    hint = CacheHint(persistence=CachePersistence.SHORT)

    reported: dict[str, tuple[int, int]] = {}
    for name, service in every_provider():
        first = await service.acomplete(
            prompt=ask("Say A.", instruction=head),
            max_output_tokens=BUDGET,
            cache_hint=hint,
        )
        await asyncio.sleep(2)
        second = await service.acomplete(
            prompt=ask("Say B.", instruction=head),
            max_output_tokens=BUDGET,
            cache_hint=hint,
        )
        assert first.usage and second.usage
        reported[name] = (
            second.usage.cached_input_tokens,
            second.usage.cache_write_tokens,
        )

    # Gemini exposes no per-request cache control, so it is the one provider
    # allowed to report nothing; the rest must show a write or a read.
    for name, (read, written) in reported.items():
        if name == "gemini":
            continue
        assert read > 0 or written > 0, f"{name} reported no cache activity: {reported}"


async def test_a_cache_hint_never_changes_the_answer(make_settings: Any) -> None:
    """Asking to cache must not alter what the prompt means."""
    configure_all(make_settings)
    service = AnthropicService(config_for(ProviderType.ANTHROPIC, ANTHROPIC_MODEL))
    plain = await service.acomplete(
        prompt=ask("Capital of Japan?"), max_output_tokens=BUDGET
    )
    hinted = await service.acomplete(
        prompt=ask("Capital of Japan?"),
        max_output_tokens=BUDGET,
        cache_hint=CacheHint(),
    )
    assert plain.content and hinted.content
    assert "tokyo" in plain.content.lower()
    assert "tokyo" in hinted.content.lower()


@pytest.mark.xfail(
    reason="Gemini 3.x refuses a replayed functionCall that carries no "
    "thought_signature, and neither StoredToolCall nor the Gemini adapter "
    "keeps one, so multi-turn tool calling cannot work there",
    strict=False,
)
async def test_a_tool_call_round_trips_through_a_second_turn(
    make_settings: Any,
) -> None:
    """The multi-step shape a real agent uses, end to end.

    The model asks for a tool, the caller runs it, the answer goes back as a
    tool message, and the model uses it. Gemini matches a tool result by
    function name rather than by id, so this is the path that proves the
    adapter's name resolution against the real service.
    """
    configure_all(make_settings)
    for name, service in every_provider():
        first = await service.acomplete(
            prompt=ask("What is the weather in Cairo?", instruction="Use tools."),
            tools=[weather_tool()],
            max_output_tokens=BUDGET,
        )
        calls = first.tool_calls
        assert calls, f"{name} asked for no tool: finish={first.finish_reason}"
        call = calls[0]
        assert call.function.name == "get_weather", name
        assert "cairo" in call.function.arguments.lower(), name

        from uuid import UUID

        from base.models.view_models.chat_history import StoredMessage

        conversation = UUID(int=7)
        history = [
            StoredMessage(
                conversation_id=conversation,
                sequence=1,
                role=MessageRole.USER,
                content="What is the weather in Cairo?",
            ),
            StoredMessage(
                conversation_id=conversation,
                sequence=2,
                role=MessageRole.ASSISTANT,
                content=first.content or "",
                tool_calls=calls,
            ),
            StoredMessage(
                conversation_id=conversation,
                sequence=3,
                role=MessageRole.TOOL,
                content="It is 21 degrees and clear in Cairo.",
                tool_call_id=call.id,
            ),
        ]
        second = await service.acomplete(
            messages=history, tools=[weather_tool()], max_output_tokens=BUDGET
        )
        assert second.content, f"{name} said nothing after the tool result"
        assert "21" in second.content, (
            f"{name} ignored the tool result: {second.content!r}"
        )


async def test_a_conversation_carries_its_own_history(make_settings: Any) -> None:
    """Several turns, with the answer depending on an earlier one."""
    configure_all(make_settings)
    from uuid import UUID

    from base.models.view_models.chat_history import StoredMessage

    conversation = UUID(int=11)
    history = [
        StoredMessage(
            conversation_id=conversation,
            sequence=1,
            role=MessageRole.USER,
            content="My favourite number is 42. Remember it.",
        ),
        StoredMessage(
            conversation_id=conversation,
            sequence=2,
            role=MessageRole.ASSISTANT,
            content="Noted, your favourite number is 42.",
        ),
        StoredMessage(
            conversation_id=conversation,
            sequence=3,
            role=MessageRole.USER,
            content="What is my favourite number? Reply with digits only.",
        ),
    ]
    for name, service in every_provider():
        result = await service.acomplete(messages=history, max_output_tokens=BUDGET)
        assert result.content and "42" in result.content, (
            f"{name} lost the history: {result.content!r}"
        )


async def test_unicode_survives_the_round_trip(make_settings: Any) -> None:
    """Content is carried, not re-encoded, whatever is in it."""
    configure_all(make_settings)
    service = AnthropicService(config_for(ProviderType.ANTHROPIC, ANTHROPIC_MODEL))
    result = await service.acomplete(
        prompt=ask("Repeat exactly: مرحبا 🌍", instruction="Repeat the text exactly."),
        max_output_tokens=BUDGET,
    )
    assert result.content
    assert "مرحبا" in result.content or "🌍" in result.content


async def test_both_provider_types_reach_a_gateway(make_settings: Any) -> None:
    """A gateway publishes its own path, and must not have one appended.

    Configuring the same endpoint as Azure used to rewrite it to `/openai/v1`
    and answer 401. Both spellings must now reach the same service.
    """
    configure_all(make_settings)
    for provider in (ProviderType.OPENAI, ProviderType.AZURE_OPENAI):
        config = config_for(provider, OPENAI_MODEL)
        config.supports_responses_api = False
        result = await OpenAIService(config).acomplete(
            prompt=ask("Capital of France?"), max_output_tokens=BUDGET
        )
        assert result.content and "paris" in result.content.lower(), provider


async def test_an_unknown_model_becomes_a_domain_error(make_settings: Any) -> None:
    """A provider's refusal must arrive as this library's error, not the SDK's."""
    configure_all(make_settings)
    from base.models.view_models.exceptions.llm import LLMError

    config = config_for(ProviderType.OPENAI, "no.such.model.anywhere")
    config.supports_responses_api = False
    with pytest.raises(LLMError):
        await OpenAIService(config).acomplete(prompt=ask("hello"), max_output_tokens=64)


async def test_a_rejected_credential_becomes_a_domain_error(
    make_settings: Any,
) -> None:
    """The same, for the one failure every deployment eventually hits."""
    configure_all(make_settings)
    from base.models.view_models.exceptions.llm import LLMError

    config = config_for(ProviderType.OPENAI, OPENAI_MODEL)
    config.api_key = "definitely-not-a-valid-key"
    config.supports_responses_api = False
    with pytest.raises(LLMError):
        await OpenAIService(config).acomplete(prompt=ask("hello"), max_output_tokens=64)


@pytest.mark.xfail(
    reason="this gateway accepts Anthropic's output_config.format without "
    "enforcing it, so the reply comes back as fenced markdown and "
    "model_validate_json rejects it",
    strict=False,
)
async def test_structured_output_returns_the_schema(make_settings: Any) -> None:
    """Every provider must parse into the model the caller asked for."""
    configure_all(make_settings)
    for name, service in every_provider():
        parsed = await service.aparse(
            Capital,
            prompt=ask(
                "What is the capital of Japan, and which country is it in?",
                instruction="Answer as JSON.",
            ),
            max_output_tokens=BUDGET,
        )
        assert isinstance(parsed.parsed, Capital), name
        assert "tokyo" in parsed.parsed.city.lower(), f"{name}: {parsed.parsed}"
        assert "japan" in parsed.parsed.country.lower(), f"{name}: {parsed.parsed}"


async def test_several_calls_can_run_concurrently(make_settings: Any) -> None:
    """Services are keyed singletons, so one must serve many callers at once."""
    configure_all(make_settings)
    service = AnthropicService(config_for(ProviderType.ANTHROPIC, ANTHROPIC_MODEL))
    questions = ["Capital of France?", "Capital of Italy?", "Capital of Japan?"]
    results = await asyncio.gather(
        *(
            service.acomplete(prompt=ask(question), max_output_tokens=BUDGET)
            for question in questions
        )
    )
    answers = " ".join((r.content or "").lower() for r in results)
    assert "paris" in answers and "rome" in answers and "tokyo" in answers
