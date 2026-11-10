"""Finding personal data by shape, and replacing it before it travels further."""

import re
from collections.abc import Callable
from re import Match, Pattern
from typing import ClassVar

from base.models.definitions.guardrails import GuardrailStage
from base.models.view_models.guardrails import GuardrailVerdict
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsGuardrailConfig
from base.services.guardrails import AbstractGuardrail

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+\.[\w.-]+\b")
CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,24}\d(?![\w-])")
IBAN = re.compile(
    r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b", re.IGNORECASE
)
PHONE = re.compile(
    r"(?<![\w#+(-])(?:[+(]|(?<!\d[.:]))\d[\d\s().-]{5,16}\d(?![\w-]|[:.]\d)"
)
PHONE_WORD = re.compile(
    r"\b(?:call|phone|telephone|tel|mobile|mob|cell|whatsapp|text|sms|fax|ring|dial)\b",
    re.IGNORECASE,
)
GROUP = re.compile(r"[^ -]+")
CARD_MAX_GROUPS = 5
IBAN_MAX_GROUPS = 9
CARD_GROUPINGS = frozenset({(4, 4, 4, 4), (4, 4, 4, 4, 3), (4, 6, 5), (4, 6, 4)})
IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
DATE_LIKE = re.compile(
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$"
)

type Locate = Callable[[Match[str]], tuple[int, int] | None]


class PatternPiiGuardrail(AbstractGuardrail):
    """Redacts emails, phone numbers, payment cards and IBANs, at every stage."""

    PLACEHOLDER: ClassVar[str] = "[redacted:{category}]"
    PHONE_WORD_REACH: ClassVar[int] = 24

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
        text = self._replace(text, IBAN, "iban", found, self._iban)
        text = self._replace(text, CARD, "card", found, self._card)
        text = self._replace(text, PHONE, "phone", found, self._phone)
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
        locate: Locate | None = None,
    ) -> str:
        """Swap the part of each match `locate` points at for a placeholder."""

        def swap(match: Match[str]) -> str:
            value = match.group(0)
            span = locate(match) if locate else (0, len(value))
            if span is None:
                return value
            found.append(category)
            start, end = span
            placeholder = self.PLACEHOLDER.format(category=category)
            return value[:start] + placeholder + value[end:]

        return pattern.sub(swap, text)

    @staticmethod
    def _window(
        value: str, valid: Callable[[str], bool], most: int
    ) -> tuple[int, int] | None:
        """The longest run of at most `most` whole groups that `valid` accepts."""
        groups = [match.span() for match in GROUP.finditer(value)]
        for size in range(min(len(groups), most), 0, -1):
            for first in range(len(groups) - size + 1):
                start, end = groups[first][0], groups[first + size - 1][1]
                if valid(value[start:end]):
                    return start, end
        return None

    def _card(self, match: Match[str]) -> tuple[int, int] | None:
        return self._window(match.group(0), self._is_card, CARD_MAX_GROUPS)

    def _iban(self, match: Match[str]) -> tuple[int, int] | None:
        return self._window(match.group(0), self._is_iban, IBAN_MAX_GROUPS)

    def _phone(self, match: Match[str]) -> tuple[int, int] | None:
        return (0, len(match.group(0))) if self._is_phone(match) else None

    @staticmethod
    def _digits(value: str) -> str:
        return re.sub(r"\D", "", value)

    def _is_card(self, value: str) -> bool:
        """Grouped the way cards are printed, and passing Luhn."""
        groups = tuple(len(group) for group in re.findall(r"\d+", value))
        if len(groups) > 1 and groups not in CARD_GROUPINGS:
            return False
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
        if not 15 <= len(candidate) <= 34 or not IBAN.fullmatch(candidate):
            return False
        rearranged = candidate[4:] + candidate[:4]
        converted = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
        if not converted.isdigit():
            return False
        return int(converted) % 97 == 1

    def _is_phone(self, match: Match[str]) -> bool:
        """Grouped numbers count; bare or dotted runs need a phone word before them."""
        value = match.group(0)
        if DATE_LIKE.match(value.strip("()")) or IPV4.match(value):
            return False
        groups = re.findall(r"\d+", value)
        if len(groups) >= 3 and all(len(group) == 4 for group in groups):
            return False
        digits = self._digits(value)
        if not 7 <= len(digits) <= 15:
            return False
        if value.startswith("+") or any(sep in value for sep in " -()"):
            return True
        if "." not in value and len(digits) not in (10, 11):
            return False
        reach = match.string[
            max(0, match.start() - self.PHONE_WORD_REACH) : match.start()
        ]
        return PHONE_WORD.search(reach) is not None


__all__ = ["PatternPiiGuardrail"]
