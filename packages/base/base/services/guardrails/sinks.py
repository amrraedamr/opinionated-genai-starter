"""The two places redacted text still escapes.

A guardrail cleans what the model is shown. It does not clean what the service
writes down about it. A prompt refused for containing a card number is
routinely logged in the same breath as the refusal, and the span that recorded
the call carries the prompt as an attribute, which is then shipped to whoever
runs the collector.

Both of those outlive the request and are read by more people than the model
ever was, so the same patterns are applied again on the way out.

Neither sink is on by default. Both cost a regular expression pass over every
line and every attribute, and a service that has not asked for guardrails has
not agreed to pay that.
"""

from typing import TYPE_CHECKING, Final

from base.models.definitions.guardrails import GuardrailStage
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsGuardrailConfig

if TYPE_CHECKING:
    from base.services.guardrails.pii import PatternPiiGuardrail

_guardrail: "PatternPiiGuardrail | None" = None
_resolved = False

MAX_ATTRIBUTE_LENGTH: Final = 8_192
"""Past this an attribute is not prose and scanning it costs more than it
protects. A prompt that long is already being truncated by the exporter."""


def pii_redactor() -> "PatternPiiGuardrail | None":
    """The configured guardrail, or None when this service did not ask for it.

    Resolved once and remembered, including the answer that there is none.
    Logging is configured before settings exist in some entry points, so this
    has to survive being asked too early rather than raise inside a log call
    and lose the line it was trying to write.
    """
    global _guardrail, _resolved
    if _resolved:
        return _guardrail
    try:
        settings = settings_for(SupportsGuardrailConfig)
    except Exception:
        return None
    _resolved = True
    if settings.guardrails_enabled and settings.pii_redaction_enabled:
        from base.services.guardrails.pii import PatternPiiGuardrail

        _guardrail = PatternPiiGuardrail(settings)
    return _guardrail


def reset_pii_redactor() -> None:
    """Forget the answer, so a test can change the settings behind it."""
    global _guardrail, _resolved
    _guardrail, _resolved = None, False


def redact_pii(text: str) -> str:
    """Replace personal data in `text`, or hand it back untouched."""
    guardrail = pii_redactor()
    if guardrail is None or not text:
        return text
    verdict = guardrail.check(text, GuardrailStage.OUTPUT)
    return verdict.content if verdict.content is not None else text


__all__ = [
    "MAX_ATTRIBUTE_LENGTH",
    "pii_redactor",
    "redact_pii",
    "reset_pii_redactor",
]
