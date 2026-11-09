"""Choosing which provider holds an open session.

Separate from the LLM factory because the set is different: a provider with a
perfectly good completions API may have no realtime API at all, and Anthropic
currently does not. Selecting by the same enum would let a caller ask for a
session no amount of configuration can produce.
"""

from typing import TYPE_CHECKING

from base.models.definitions.model import ProviderType
from base.models.view_models.model import ModelConfigurations
from base.models.view_models.realtime_session import RealtimeSessionConfig
from base.services.llm.realtime import AbstractRealtimeSession

if TYPE_CHECKING:
    from base.models.view_models.realtime_session import ToolSchema

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
    provider: ProviderType,
    session: RealtimeSessionConfig,
    config: ModelConfigurations | None = None,
    tool_schemas: "list[ToolSchema] | None" = None,
) -> AbstractRealtimeSession:
    """Construct a session for the provider asked for.

    Each adapter imports lazily, so a service using Gemini needs no `openai`
    extra and the reverse. Dispatch is exhaustive rather than defaulting:
    falling back to a provider the caller did not name would open a session
    against the wrong model and bill it silently.
    """
    match provider:
        case ProviderType.OPENAI | ProviderType.AZURE_OPENAI:
            from base.services.llm.realtime.openai import OpenAIRealtimeSession

            return OpenAIRealtimeSession(session, config, tool_schemas)
        case ProviderType.GEMINI:
            from base.services.llm.realtime.gemini import GeminiLiveSession

            return GeminiLiveSession(session, config, tool_schemas)
        case _:
            raise ValueError(
                f"{provider.value!r} has no realtime API. "
                f"Providers that do: "
                f"{', '.join(p.value for p in REALTIME_PROVIDERS)}."
            )


__all__ = ["REALTIME_PROVIDERS", "build_realtime_session"]
