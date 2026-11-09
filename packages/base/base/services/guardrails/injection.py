"""Catching text that is trying to talk to the model rather than answer it.

This runs on tool results only, and that restriction is the point. A user
saying "ignore previous instructions" is a nuisance the model can refuse. The
same sentence arriving inside a tool result is different: it is text the model
was told to trust, from a third party, and a forked repository will point this
at MCP servers nobody here reviewed.

Heuristics cannot catch everything written to steer a model. They catch the
shapes that actually appear, and raise the cost of the rest.
"""

import re
from typing import ClassVar

from base.models.definitions.guardrails import GuardrailStage
from base.models.view_models.guardrails import GuardrailVerdict
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsGuardrailConfig
from base.services.guardrails import AbstractGuardrail
from base.tools.sanitize import has_invisible

OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
    r"(?:previous|prior|above|earlier|system|all)\b[^.\n]{0,20}"
    r"\b(?:instruction|instructions|prompt|prompts|rule|rules|context)\b",
    re.IGNORECASE,
)
IDENTITY = re.compile(
    r"\byou\s+are\s+now\b|\bact\s+as\s+(?:a|an|the)\b|\bpretend\s+(?:to\s+be|you)\b"
    r"|\bfrom\s+now\s+on\s+you\b|\bnew\s+instructions\b",
    re.IGNORECASE,
)
ROLE_MARKER = re.compile(
    r"<\|im_(?:start|end)\|>|<\|endoftext\|>|\[/?INST\]|<<SYS>>"
    r"|^\s*(?:system|assistant)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
EXFILTRATION = re.compile(
    r"\b(?:reveal|print|repeat|show|output|disclose)\b[^.\n]{0,30}"
    r"\b(?:system\s+prompt|your\s+instructions|the\s+prompt|api\s+key|secret)\b",
    re.IGNORECASE,
)


class InjectionHeuristicGuardrail(AbstractGuardrail):
    """Blocks tool output that reads as an instruction to the model."""

    SIGNALS: ClassVar[dict[str, re.Pattern[str]]] = {
        "instruction-override": OVERRIDE,
        "identity-reassignment": IDENTITY,
        "role-marker": ROLE_MARKER,
        "exfiltration": EXFILTRATION,
    }

    def __init__(self, config: SupportsGuardrailConfig | None = None) -> None:
        self._config = config or settings_for(SupportsGuardrailConfig)

    def _name(self) -> str:
        return "injection-heuristic"

    def _stages(self) -> frozenset[GuardrailStage]:
        """Tool results only. A user is allowed to say anything to the model."""
        return frozenset({GuardrailStage.TOOL_RESULT})

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        del stage
        return self._verdict(content)

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        del stage
        return self._verdict(content)

    def _verdict(self, content: str) -> GuardrailVerdict:
        if not content or not self._config.injection_block_enabled:
            return GuardrailVerdict.allow()

        if has_invisible(content):
            return GuardrailVerdict.block(
                "Tool result contains invisible characters.",
                category="invisible-characters",
            )

        for category, pattern in self.SIGNALS.items():
            if pattern.search(content):
                return GuardrailVerdict.block(
                    "Tool result reads as an instruction to the model.",
                    category=category,
                )
        return GuardrailVerdict.allow()


__all__ = ["InjectionHeuristicGuardrail"]
