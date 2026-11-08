"""Where a guardrail looks, and what it can decide."""

from enum import StrEnum


class GuardrailStage(StrEnum):
    """The point in a turn a guardrail is checking."""

    INPUT = "input"
    OUTPUT = "output"
    TOOL_RESULT = "tool_result"


class GuardrailOutcome(StrEnum):
    """What a guardrail decided about the content it was shown."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"
