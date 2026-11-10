"""Where a guardrail looks, and what it can decide."""

from enum import StrEnum
from typing import Final

GUARDRAIL_CACHE_NAMESPACE: Final = "guardrails:report"
GUARDRAIL_CACHE_TTL_SECONDS: Final = 86_400

STREAM_CUT_ATTEMPTS: Final = 12
"""Whitespace positions tried per chunk when looking for a cut that splits no value."""
STREAM_HELD_LIMIT: Final = 2_048
"""Past this much held text with no clean cut, the checked buffer is sent whole."""

RETRACTION_CODE: Final = 422
RETRACTION_TEXT: Final = "The reply was withdrawn by a content policy."

BLOCKED_HISTORY_PLACEHOLDER: Final = (
    "[An earlier message was withheld because it contained content "
    "this service does not process.]"
)


class GuardrailStage(StrEnum):
    """The point in a turn a guardrail is checking."""

    INPUT = "input"
    OUTPUT = "output"
    OUTPUT_DELTA = "output_delta"
    """Part of a reply still streaming; for checks that can judge a fragment."""
    TOOL_RESULT = "tool_result"


class GuardrailOutcome(StrEnum):
    """What a guardrail decided about the content it was shown."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"
