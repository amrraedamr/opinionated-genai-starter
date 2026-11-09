"""Choosing a provider to hold a session.

The set is not the set of providers that answer completions, and the gap is
the point: a caller can ask for a session only where one exists.
"""

import pytest

from base.models.definitions.model import ProviderType
from base.models.definitions.realtime_session import RealtimeModality
from base.models.view_models.realtime_session import RealtimeSessionConfig
from base.services.configuration import settings_for
from base.services.configuration.mixins import RealtimeSettingsMixin
from base.services.configuration.protocols import SupportsRealtimeConfig
from base.services.llm.realtime import AbstractRealtimeSession
from base.services.llm.realtime.factory import (
    REALTIME_PROVIDERS,
    build_realtime_session,
    session_from,
)
from base.services.llm.realtime.gemini import GeminiLiveSession
from base.services.llm.realtime.openai import OpenAIRealtimeSession
from tests.conftest import MakeSettings

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


# Falling back to settings


def test_a_caller_with_nothing_to_say_still_gets_a_session(
    make_settings: MakeSettings,
) -> None:
    """Which model answers is a deployment's business, not a handler's.

    Without this a route has to read settings and assemble a session itself,
    which puts configuration in the one place that should only be wiring.
    """
    make_settings(RealtimeSettingsMixin, realtime_provider="gemini")
    assert isinstance(build_realtime_session(), GeminiLiveSession)


def test_the_configured_model_is_the_one_opened(
    make_settings: MakeSettings,
) -> None:
    make_settings(RealtimeSettingsMixin, realtime_model="gpt-realtime-mini")
    assert build_realtime_session().model == "gpt-realtime-mini"


def test_a_session_speaks_when_the_deployment_asked_for_audio(
    make_settings: MakeSettings,
) -> None:
    make_settings(RealtimeSettingsMixin, realtime_audio_enabled=True)
    built = session_from(settings_for(SupportsRealtimeConfig))
    assert built.modalities == [RealtimeModality.AUDIO]


def test_a_session_stays_typed_when_it_did_not(
    make_settings: MakeSettings,
) -> None:
    """Cheaper, and sometimes all a service wants."""
    make_settings(RealtimeSettingsMixin, realtime_audio_enabled=False)
    built = session_from(settings_for(SupportsRealtimeConfig))
    assert built.modalities == [RealtimeModality.TEXT]


def test_an_empty_voice_is_left_unset_rather_than_sent_blank(
    make_settings: MakeSettings,
) -> None:
    make_settings(RealtimeSettingsMixin, realtime_voice="")
    assert session_from(settings_for(SupportsRealtimeConfig)).voice is None


def test_an_explicit_provider_is_not_overridden_by_settings(
    make_settings: MakeSettings,
) -> None:
    """A caller that named one means it, whatever the deployment prefers."""
    make_settings(RealtimeSettingsMixin, realtime_provider="gemini")
    built = build_realtime_session(ProviderType.OPENAI, CONFIG)
    assert isinstance(built, OpenAIRealtimeSession)


def test_a_fully_explicit_call_needs_no_settings_at_all() -> None:
    """So a worker with no realtime settings can still open one deliberately."""
    assert build_realtime_session(ProviderType.GEMINI, CONFIG) is not None
