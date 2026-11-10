"""Assembling the pipeline a service is configured to run.

Off unless asked. Enabling guardrails changes what the model is shown and what
a caller gets back, which is not a thing to switch on under someone without
their say-so — the same reasoning that makes tenancy opt-in.

Within that, PII redaction and injection heuristics are on because they are
local and cost nothing per check. Moderation is off because it bills a provider
call every time, and a starter that quietly spends money is a bad starter.
"""

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

    PII runs before moderation so a flagged message is not shipped to a third
    party with the personal data still in it. Injection runs last because it
    only looks at tool results, which the other two have already cleaned.

    An empty pipeline is a valid answer, not an error: `applies_to` is false
    for every stage, so callers skip the work entirely.
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
    """The composed settings, or the mixin's own defaults.

    Both failures mean the same thing here: nobody asked for guardrails. A
    service may have composed no settings at all, or settings without this
    mixin, and either way the answer is the defaults rather than a crash on a
    feature that is off anyway.
    """
    try:
        return settings_for(SupportsGuardrailConfig)
    except (MissingSettingsError, RuntimeError):
        return GuardrailSettingsMixin()


__all__ = ["build_guarded_stream", "build_guardrail_pipeline"]
