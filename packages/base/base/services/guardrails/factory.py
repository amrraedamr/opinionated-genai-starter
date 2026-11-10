"""Assembling the pipeline a service is configured to run."""

from base.services.configuration import settings_for
from base.services.configuration.mixins.guardrails import GuardrailSettingsMixin
from base.services.configuration.protocols import SupportsGuardrailConfig
from base.services.guardrails import AbstractGuardrail
from base.services.guardrails.pipeline import GuardrailPipeline
from base.services.guardrails.streaming import GuardedStream
from base.tools.errors import MissingSettingsError


def build_guardrail_pipeline(
    config: SupportsGuardrailConfig | None = None,
) -> GuardrailPipeline:
    """The configured guardrails, in the order they should run.

    PII runs before moderation so personal data is not sent to the moderation provider.
    """
    settings = config or _configured()
    if not settings.guardrails_enabled:
        return GuardrailPipeline()

    guardrails: list[AbstractGuardrail] = []
    if settings.pii_redaction_enabled:
        from base.services.guardrails.pii import PatternPiiGuardrail

        guardrails.append(PatternPiiGuardrail(settings))
    if settings.moderation_enabled:
        from base.services.guardrails.llm.openai import OpenAIModerationGuardrail

        guardrails.append(OpenAIModerationGuardrail(settings))
    if settings.injection_block_enabled:
        from base.services.guardrails.injection import InjectionHeuristicGuardrail

        guardrails.append(InjectionHeuristicGuardrail(settings))

    return GuardrailPipeline(*guardrails)


def build_guarded_stream(
    guardrails: GuardrailPipeline, config: SupportsGuardrailConfig | None = None
) -> GuardedStream:
    """A guard for one streamed reply, holding back the configured tail."""
    return GuardedStream(guardrails, tail=(config or _configured()).pii_stream_tail)


def _configured() -> SupportsGuardrailConfig:
    """The composed settings, or the mixin's defaults when settings are unconfigured or lack the mixin."""
    try:
        return settings_for(SupportsGuardrailConfig)
    except (MissingSettingsError, RuntimeError):
        return GuardrailSettingsMixin()


__all__ = ["build_guarded_stream", "build_guardrail_pipeline"]
