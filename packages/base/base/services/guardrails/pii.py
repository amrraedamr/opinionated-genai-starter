"""Finding personal data by shape, and replacing it before it travels further.

Patterns alone over-report badly: an order number is a run of digits, and so is
a card. Everything here that can be checked arithmetically is — cards by Luhn,
IBANs by mod-97 — and the rest is narrowed by what sits around the match. A
guardrail that cries wolf gets turned off, which protects nothing.
"""

import re
from re import Pattern
from typing import ClassVar

from base.models.definitions.guardrails import GuardrailStage
from base.models.view_models.guardrails import GuardrailVerdict
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsGuardrailConfig
from base.services.guardrails import AbstractGuardrail

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
PHONE = re.compile(r"(?<![\w#-])\+?\d[\d\s().-]{5,16}\d(?![\w-])")
DATE_LIKE = re.compile(
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$"
)


class PatternPiiGuardrail(AbstractGuardrail):
    """Redacts emails, phone numbers, payment cards and IBANs.

    Runs at every stage: the same number is a problem arriving from a user, in
    a reply, and in whatever a tool hands back.
    """

    PLACEHOLDER: ClassVar[str] = "[redacted:{category}]"

    def __init__(self, config: SupportsGuardrailConfig | None = None) -> None:
        self._config = config or settings_for(SupportsGuardrailConfig)
        self._extra: dict[str, Pattern[str]] = {
            name: re.compile(pattern)
            for name, pattern in self._config.pii_extra_patterns.items()
        }

    def _name(self) -> str:
        return "pattern-pii"

    def _stages(self) -> frozenset[GuardrailStage]:
        return frozenset(GuardrailStage)

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        del stage
        return self._redacted(content)

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        del stage
        return self._redacted(content)

    def _redacted(self, content: str) -> GuardrailVerdict:
        """Replace every recognised value, or allow the text untouched."""
        if not content:
            return GuardrailVerdict.allow()

        found: list[str] = []
        text = content
        text = self._replace(text, EMAIL, "email", found)
        text = self._replace(text, IBAN, "iban", found, self._is_iban)
        text = self._replace(text, CARD, "card", found, self._is_card)
        text = self._replace(text, PHONE, "phone", found, self._is_phone)
        for name, pattern in self._extra.items():
            text = self._replace(text, pattern, name, found)

        if not found:
            return GuardrailVerdict.allow()
        return GuardrailVerdict.redact(text, category=",".join(sorted(set(found))))

    def _replace(
        self,
        text: str,
        pattern: Pattern[str],
        category: str,
        found: list[str],
        accept: object = None,
    ) -> str:
        """Swap matches `accept` agrees with for a placeholder naming the kind."""

        def swap(match: re.Match[str]) -> str:
            value = match.group(0)
            if callable(accept) and not accept(value):
                return value
            found.append(category)
            return self.PLACEHOLDER.format(category=category)

        return pattern.sub(swap, text)

    @staticmethod
    def _digits(value: str) -> str:
        return re.sub(r"\D", "", value)

    def _is_card(self, value: str) -> bool:
        """Luhn is what separates a card from any other run of digits."""
        digits = self._digits(value)
        if not 13 <= len(digits) <= 19:
            return False
        total = 0
        for index, char in enumerate(reversed(digits)):
            digit = int(char)
            if index % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0

    @staticmethod
    def _is_iban(value: str) -> bool:
        """Mod-97 over the rearranged account number, which is the standard."""
        candidate = value.replace(" ", "").upper()
        if not 15 <= len(candidate) <= 34:
            return False
        rearranged = candidate[4:] + candidate[:4]
        converted = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
        if not converted.isdigit():
            return False
        return int(converted) % 97 == 1

    def _is_phone(self, value: str) -> bool:
        """A run of digits is only a phone number if it is shaped like one.

        A bare eight-digit run is far more often an order or invoice number, so
        bare runs are only accepted at the lengths real numbers actually take.
        """
        trimmed = value.strip()
        if DATE_LIKE.match(trimmed):
            return False
        digits = self._digits(trimmed)
        if not 7 <= len(digits) <= 15:
            return False
        if trimmed.startswith("+"):
            return True
        if any(separator in trimmed for separator in " .-()"):
            return True
        return len(digits) in (10, 11)


__all__ = ["PatternPiiGuardrail"]
