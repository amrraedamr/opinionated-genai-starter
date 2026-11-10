"""Choosing which provider holds an open session.

Separate from the LLM factory because the set is different: a provider with a
perfectly good completions API may have no realtime API at all, and Anthropic
currently does not. Selecting by the same enum would let a caller ask for a
session no amount of configuration can produce.
"""

from base.models.definitions.model import ProviderType
from base.models.definitions.realtime_session import RealtimeModality
from base.models.view_models.model import ModelConfigurations
from base.models.view_models.realtime_session import RealtimeSessionConfig
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsRealtimeConfig
from base.services.llm.realtime import AbstractRealtimeSession
from base.services.tool_calling import Toolbox

REALTIME_PROVIDERS = (
    ProviderType.OPENAI,
    ProviderType.AZURE_OPENAI,
    ProviderType.GEMINI,
)
"""Which providers can hold a session at all.

Anthropic is absent rather than stubbed. A stub that raises on first use
reads as support until something tries it; an absence is answerable before a
caller writes the code around it.
"""


def build_realtime_session(
    provider: ProviderType | None = None,
    session: RealtimeSessionConfig | None = None,
    config: ModelConfigurations | None = None,
    toolbox: Toolbox | None = None,
) -> AbstractRealtimeSession:
    """Construct the session this service is configured for.

    Both the provider and the session shape fall back to settings, the way
    `build_broker` does, so a caller that has nothing to say about either
    asks for a session and gets one. A route is then wiring, not
    configuration: deciding which model answers is a deployment's business,
    not a handler's.

    Each adapter imports lazily, so a service using Gemini needs no `openai`
    extra and the reverse. Dispatch is exhaustive rather than defaulting:
    falling back to a provider the caller did not name would open a session
    against the wrong model and bill it silently.
    """
    if provider is None or session is None:
        settings = settings_for(SupportsRealtimeConfig)
        provider = provider or ProviderType(settings.realtime_provider)
        session = session or session_from(settings)

    match provider:
        case ProviderType.OPENAI | ProviderType.AZURE_OPENAI:
            from base.services.llm.realtime.openai import OpenAIRealtimeSession

            return OpenAIRealtimeSession(session, config, toolbox, provider)
        case ProviderType.GEMINI:
            from base.services.llm.realtime.gemini import GeminiLiveSession

            return GeminiLiveSession(session, config, toolbox)
        case _:
            raise ValueError(
                f"{provider.value!r} has no realtime API. "
                f"Providers that do: "
                f"{', '.join(p.value for p in REALTIME_PROVIDERS)}."
            )


def session_from(settings: SupportsRealtimeConfig) -> RealtimeSessionConfig:
    """The session a deployment asked for, in neutral terms."""
    return RealtimeSessionConfig(
        model=settings.realtime_model,
        modalities=[RealtimeModality.AUDIO]
        if settings.realtime_audio_enabled
        else [RealtimeModality.TEXT],
        voice=settings.realtime_voice or None,
        instructions=settings.realtime_instructions or None,
    )


__all__ = ["REALTIME_PROVIDERS", "build_realtime_session", "session_from"]
