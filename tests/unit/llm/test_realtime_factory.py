"""Choosing a provider to hold a session.

The set is not the set of providers that answer completions, and the gap is
the point: a caller can ask for a session only where one exists.
"""

import pytest

from base.models.definitions.model import ProviderType
from base.models.view_models.realtime_session import RealtimeSessionConfig
from base.services.llm.realtime import AbstractRealtimeSession
from base.services.llm.realtime.factory import (
    REALTIME_PROVIDERS,
    build_realtime_session,
)
from base.services.llm.realtime.gemini import GeminiLiveSession
from base.services.llm.realtime.openai import OpenAIRealtimeSession

CONFIG = RealtimeSessionConfig(model="m")


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (ProviderType.OPENAI, OpenAIRealtimeSession),
        (ProviderType.AZURE_OPENAI, OpenAIRealtimeSession),
        (ProviderType.GEMINI, GeminiLiveSession),
    ],
)
def test_each_supported_provider_builds_its_own_session(
    provider: ProviderType, expected: type
) -> None:
    assert isinstance(build_realtime_session(provider, CONFIG), expected)


def test_every_built_session_satisfies_the_seam() -> None:
    for provider in REALTIME_PROVIDERS:
        built = build_realtime_session(provider, CONFIG)
        assert isinstance(built, AbstractRealtimeSession)


def test_a_provider_without_a_realtime_api_is_refused() -> None:
    """Anthropic answers completions and holds no session.

    Refusing here is the whole reason this is not the LLM factory: selecting
    by the same enum would let a caller ask for something no amount of
    configuration can produce.
    """
    with pytest.raises(ValueError, match="no realtime API"):
        build_realtime_session(ProviderType.ANTHROPIC, CONFIG)


def test_the_refusal_names_the_providers_that_do() -> None:
    """So the answer arrives with the error rather than after a search."""
    with pytest.raises(ValueError, match="gemini"):
        build_realtime_session(ProviderType.OLLAMA, CONFIG)


def test_anthropic_is_absent_rather_than_stubbed() -> None:
    """A stub that raises on first use reads as support until something tries."""
    assert ProviderType.ANTHROPIC not in REALTIME_PROVIDERS


def test_tool_schemas_reach_the_session() -> None:
    schemas = [{"name": "lookup"}]
    built = build_realtime_session(ProviderType.GEMINI, CONFIG, tool_schemas=schemas)
    assert built._tool_schemas == schemas  # type: ignore[attr-defined]
