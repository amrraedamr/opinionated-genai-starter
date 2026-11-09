"""Finding personal data without crying wolf.

Half of these are things that must *not* be redacted. A guardrail that mangles
order numbers and dates gets switched off, and a switched-off guardrail
protects nothing, so the false positives are pinned as carefully as the true
ones.
"""

import pytest

from base.models.definitions.guardrails import GuardrailOutcome, GuardrailStage
from base.services.configuration.mixins import GuardrailSettingsMixin
from base.services.guardrails.pii import PatternPiiGuardrail
from tests.conftest import MakeSettings

STAGES = list(GuardrailStage)


@pytest.fixture
def guardrail(make_settings: MakeSettings) -> PatternPiiGuardrail:
    make_settings(GuardrailSettingsMixin)
    return PatternPiiGuardrail()


def redacted(guardrail: PatternPiiGuardrail, text: str) -> str:
    """The text as it survives the guardrail."""
    verdict = guardrail.check(text, GuardrailStage.INPUT)
    return verdict.content if verdict.content is not None else text


@pytest.mark.parametrize(
    "text",
    [
        "write to amr@example.com please",
        "amr.raed+tag@sub.example.co.uk",
        "ALL.CAPS@EXAMPLE.COM",
    ],
)
def test_an_email_is_redacted(guardrail: PatternPiiGuardrail, text: str) -> None:
    assert "[redacted:email]" in redacted(guardrail, text)


@pytest.mark.parametrize(
    "card",
    [
        "4111111111111111",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "5500005555555559",
        "378282246310005",
    ],
)
def test_a_real_card_number_is_redacted(
    guardrail: PatternPiiGuardrail, card: str
) -> None:
    """Every one of these passes Luhn, which is what makes it a card."""
    assert "[redacted:card]" in redacted(guardrail, f"card {card} on file")


def test_a_digit_run_that_fails_luhn_is_left_alone(
    guardrail: PatternPiiGuardrail,
) -> None:
    """Sixteen digits is not a card; sixteen digits that check out is."""
    assert "4111111111111112" in redacted(guardrail, "ref 4111111111111112")


@pytest.mark.parametrize(
    "iban",
    ["GB82WEST12345698765432", "DE89370400440532013000", "FR1420041010050500013M02606"],
)
def test_a_valid_iban_is_redacted(guardrail: PatternPiiGuardrail, iban: str) -> None:
    assert "[redacted:iban]" in redacted(guardrail, f"pay to {iban} today")


def test_an_iban_with_a_broken_checksum_is_left_alone(
    guardrail: PatternPiiGuardrail,
) -> None:
    """Mod-97 is the difference between an IBAN and a similar-looking string."""
    assert "GB82WEST12345698765433" in redacted(
        guardrail, "pay to GB82WEST12345698765433 today"
    )


@pytest.mark.parametrize(
    "phone",
    [
        "+44 20 7946 0958",
        "+1 (555) 123-4567",
        "020 7946 0958",
        "555-123-4567",
        "5551234567",
    ],
)
def test_a_phone_number_is_redacted(guardrail: PatternPiiGuardrail, phone: str) -> None:
    assert "[redacted:phone]" in redacted(guardrail, f"call {phone} tomorrow")


@pytest.mark.parametrize(
    "text",
    [
        "Order #12345678 shipped",
        "invoice 4567123 is overdue",
        "the meeting is on 2026-11-09",
        "due 11/09/2026 at the latest",
        "version 1.2.3 released",
        "HTTP 404 on /v1/items",
        "the total was 1250 dollars",
    ],
)
def test_ordinary_text_is_left_alone(guardrail: PatternPiiGuardrail, text: str) -> None:
    """The false positives that would make someone turn this off."""
    verdict = guardrail.check(text, GuardrailStage.INPUT)
    assert verdict.outcome == GuardrailOutcome.ALLOW, (
        f"{text!r} was redacted as {verdict.category}"
    )


def test_an_iso_date_is_not_a_phone_number(
    guardrail: PatternPiiGuardrail,
) -> None:
    """Eight digits with separators is exactly the shape of a date."""
    assert redacted(guardrail, "2026-11-09") == "2026-11-09"


def test_several_kinds_in_one_message_are_all_redacted(
    guardrail: PatternPiiGuardrail,
) -> None:
    text = "mail amr@example.com, card 4111111111111111, ring +44 20 7946 0958"
    verdict = guardrail.check(text, GuardrailStage.INPUT)
    assert verdict.outcome == GuardrailOutcome.REDACT
    assert verdict.category is not None
    assert {"email", "card", "phone"} <= set(verdict.category.split(","))


def test_what_survives_carries_no_original_value(
    guardrail: PatternPiiGuardrail,
) -> None:
    """A redaction that leaves the digits behind has done nothing."""
    survived = redacted(guardrail, "card 4111 1111 1111 1111 on file")
    assert "4111" not in survived


def test_empty_content_is_allowed(guardrail: PatternPiiGuardrail) -> None:
    assert guardrail.check("", GuardrailStage.INPUT).outcome == GuardrailOutcome.ALLOW


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_is_checked(
    guardrail: PatternPiiGuardrail, stage: GuardrailStage
) -> None:
    """The same number is a problem arriving, leaving, or from a tool."""
    assert stage in guardrail.stages
    verdict = guardrail.check("mail amr@example.com", stage)
    assert verdict.outcome == GuardrailOutcome.REDACT


async def test_both_counterparts_agree(guardrail: PatternPiiGuardrail) -> None:
    text = "card 4111111111111111"
    sync = guardrail.check(text, GuardrailStage.INPUT)
    asynchronous = await guardrail.acheck(text, GuardrailStage.INPUT)
    assert sync.content == asynchronous.content


def test_an_extra_pattern_is_honoured(make_settings: MakeSettings) -> None:
    """A deployment knows about identifiers this library cannot guess."""
    make_settings(
        GuardrailSettingsMixin,
        pii_extra_patterns={"employee-id": r"\bEMP-\d{5}\b"},
    )
    verdict = PatternPiiGuardrail().check("EMP-12345 asked", GuardrailStage.INPUT)
    assert verdict.outcome == GuardrailOutcome.REDACT
    assert verdict.content == "[redacted:employee-id] asked"


def test_the_guardrail_names_itself(guardrail: PatternPiiGuardrail) -> None:
    assert guardrail.name == "pattern-pii"
