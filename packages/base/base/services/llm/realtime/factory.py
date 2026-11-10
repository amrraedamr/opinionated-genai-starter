"""Choosing which provider holds an open session."""

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


def build_realtime_session(
    provider: ProviderType | None = None,
    session: RealtimeSessionConfig | None = None,
    config: ModelConfigurations | None = None,
    toolbox: Toolbox | None = None,
) -> AbstractRealtimeSession:
    """Construct a realtime session, taking the provider and session from settings when omitted."""
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
