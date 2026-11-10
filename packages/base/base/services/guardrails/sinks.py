"""Personal data taken off log lines and span attributes, which outlive the request."""

from typing import TYPE_CHECKING

from base.models.definitions.guardrails import GuardrailStage
from base.services.common.meta import Singleton
from base.services.configuration import is_configured, settings_for
from base.services.configuration.protocols import SupportsGuardrailConfig
from base.tools.errors import MissingSettingsError

if TYPE_CHECKING:
    from base.services.guardrails.pii import PatternPiiGuardrail


class PiiRedactor(metaclass=Singleton):
    """PII redaction for log lines and spans; off unless a service asks."""

    def __init__(self) -> None:
        self._guardrail: PatternPiiGuardrail | None = None
        self._resolved = False

    def redact(self, text: str) -> str:
        """`text` with personal data replaced, or untouched when redaction is off."""
        guardrail = self._resolve()
        if guardrail is None or not text:
            return text
        verdict = guardrail.check(text, GuardrailStage.OUTPUT)
        return verdict.content if verdict.content is not None else text

    def _resolve(self) -> "PatternPiiGuardrail | None":
        """Resolved once settings exist; earlier log lines are left alone."""
        if self._resolved or not is_configured():
            return self._guardrail
        try:
            settings = settings_for(SupportsGuardrailConfig)
        except MissingSettingsError:
            self._resolved = True
            return None
        if settings.guardrails_enabled and settings.pii_redaction_enabled:
            from base.services.guardrails.pii import PatternPiiGuardrail

            self._guardrail = PatternPiiGuardrail(settings)
        self._resolved = True
        return self._guardrail


__all__ = ["PiiRedactor"]
