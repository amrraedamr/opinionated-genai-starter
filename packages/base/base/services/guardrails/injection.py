"""Tool results that read as instructions to the model; tool results only, since users may say anything."""

import re
import unicodedata
from typing import ClassVar

from base.models.definitions.guardrails import GuardrailStage
from base.models.view_models.guardrails import GuardrailVerdict
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsGuardrailConfig
from base.services.guardrails import AbstractGuardrail
from base.tools.sanitize import TAG_BLOCK, strip_invisible

EMOJI_TAG_SEQUENCE = re.compile("\U0001f3f4[\U000e0020-\U000e007e]+\U000e007f")
WHITESPACE = re.compile(r"\s+")
LOOKALIKES = str.maketrans(
    "аеорсухіјѕԁԛԝАВЕКМНОРСТХІЈЅαοινρτυΑΒΕΗΙΚΜΝΟΡΤΥΧ",
    "aeopcyxijsdqwABEKMHOPCTXIJSaoivptuABEHIKMNOPTYX",
)

OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.]{0,40}"
    r"\b(?:previous|prior|above|earlier|preceding|original|system|your)\b[^.]{0,20}"
    r"\b(?:instructions?|prompts?|rules?|context|guidelines|directives)\b"
    r"|\b(?:ignore|disregard|forget)\s+(?:all|any|every)\s+(?:instructions|prompts|directives)\b",
    re.IGNORECASE,
)
IDENTITY = re.compile(
    r"\byou\s+are\s+now\b|\bpretend\s+(?:to\s+be|you)\b|\bfrom\s+now\s+on\s+you\b"
    r"|\bnew\s+instructions\b|(?:^|[.!?]\s)act\s+as\s+(?:a|an|the)\b",
    re.IGNORECASE,
)
ROLE_MARKER = re.compile(
    r"<\|im_(?:start|end)\|>|<\|endoftext\|>|\[/?INST\]|<<SYS>>"
    r"|^\s*(?:system|assistant)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
EXFILTRATION = re.compile(
    r"\b(?:reveal|print|repeat|show|output|disclose|leak)\b[^.]{0,30}"
    r"\b(?:system\s+prompt|your\s+(?:instructions|prompt|rules)|api\s+key|access\s+token)\b",
    re.IGNORECASE,
)


class InjectionHeuristicGuardrail(AbstractGuardrail):
    """Blocks tool output that reads as an instruction to the model, after undoing disguises."""

    SIGNALS: ClassVar[dict[str, re.Pattern[str]]] = {
        "instruction-override": OVERRIDE,
        "identity-reassignment": IDENTITY,
        "exfiltration": EXFILTRATION,
    }

    def __init__(self, config: SupportsGuardrailConfig | None = None) -> None:
        self._config = config or settings_for(SupportsGuardrailConfig)

    def _name(self) -> str:
        return "injection-heuristic"

    def _stages(self) -> frozenset[GuardrailStage]:
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

        if self._smuggles(content):
            return GuardrailVerdict.block(
                "Tool result contains invisible characters.",
                category="invisible-characters",
            )
        if ROLE_MARKER.search(content):
            return self._blocked("role-marker")

        normalised = self._normalised(content)
        for category, pattern in self.SIGNALS.items():
            if pattern.search(normalised):
                return self._blocked(category)
        return GuardrailVerdict.allow()

    @staticmethod
    def _blocked(category: str) -> GuardrailVerdict:
        return GuardrailVerdict.block(
            "Tool result reads as an instruction to the model.", category=category
        )

    @staticmethod
    def _smuggles(content: str) -> bool:
        """Tag characters outside an emoji flag encode ASCII that renders as nothing."""
        remaining = EMOJI_TAG_SEQUENCE.sub("", content)
        return any(ord(char) in TAG_BLOCK for char in remaining)

    @staticmethod
    def _normalised(content: str) -> str:
        """Zero-width characters removed, look-alike letters mapped, whitespace collapsed."""
        text = unicodedata.normalize("NFKC", strip_invisible(content))
        return WHITESPACE.sub(" ", text.translate(LOOKALIKES))


__all__ = ["InjectionHeuristicGuardrail"]
