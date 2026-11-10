"""The two places a redacted value still escapes.

A guardrail cleans what the model is shown and nothing else. These pin the
other two copies: the line the service logs about the request, and the span
attribute shipped to whoever runs the collector. Both outlive the request and
are read by more people than the model ever was.
"""

import logging

import pytest

from base.services.configuration.mixins import GuardrailSettingsMixin
from base.services.guardrails.sinks import PiiRedactor
from base.services.logging.config import JsonFormatter, RedactionFilter, redact
from tests.conftest import MakeSettings

EMAIL = "bob@example.com"
CARD = "4111 1111 1111 1111"


@pytest.fixture
def guarded(make_settings: MakeSettings) -> None:
    make_settings(GuardrailSettingsMixin, guardrails_enabled=True)


@pytest.fixture
def unguarded(make_settings: MakeSettings) -> None:
    make_settings(GuardrailSettingsMixin, guardrails_enabled=False)


# Logs


def test_an_address_does_not_reach_a_log_line(guarded: None) -> None:
    assert EMAIL not in redact(f"user {EMAIL} signed in")


def test_a_card_does_not_reach_a_log_line(guarded: None) -> None:
    assert "4111" not in redact(f"charged {CARD}")


def test_the_rest_of_the_line_survives(guarded: None) -> None:
    """A redacted log is only useful if it still says what happened."""
    line = redact(f"user {EMAIL} signed in")
    assert "signed in" in line


def test_a_credential_is_removed_whether_or_not_guardrails_run(
    unguarded: None,
) -> None:
    """Two different decisions. Nothing is gained by logging a key, ever."""
    assert "hunter2" not in redact("password=hunter2")


def test_personal_data_is_left_alone_until_a_service_asks(
    unguarded: None,
) -> None:
    """It costs a regular expression pass over every line."""
    assert EMAIL in redact(f"user {EMAIL} signed in")


def test_the_filter_redacts_the_record_itself(guarded: None) -> None:
    """The formatter is not the only reader; a handler may format it its way."""
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, f"mail {EMAIL}", None, None
    )

    RedactionFilter().filter(record)

    assert EMAIL not in str(record.msg)


def test_the_json_formatter_redacts_too(guarded: None) -> None:
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, f"mail {EMAIL}", None, None
    )
    assert EMAIL not in JsonFormatter().format(record)


def test_an_empty_line_is_handled(guarded: None) -> None:
    assert redact("") == ""


# Resolution


def test_nothing_is_redacted_before_settings_exist() -> None:
    """Logging is configured first in some entry points.

    Raising inside a log call would lose the line it was trying to write,
    which is usually the one explaining what went wrong.
    """
    assert PiiRedactor().redact(f"mail {EMAIL}") == f"mail {EMAIL}"


def test_redaction_starts_once_settings_exist(make_settings: MakeSettings) -> None:
    """A line logged before `configure` must not switch redaction off for good."""
    redactor = PiiRedactor()
    redactor.redact("early line")
    make_settings(GuardrailSettingsMixin, guardrails_enabled=True)
    assert EMAIL not in redactor.redact(f"mail {EMAIL}")


def test_a_service_without_the_mixin_is_asked_once(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every shipped service is one; asking per log line costs every line."""
    from base.services.guardrails import sinks

    make_settings()
    asked: list[object] = []
    real = sinks.settings_for

    def counting(contract: type[object]) -> object:
        asked.append(contract)
        return real(contract)

    monkeypatch.setattr(sinks, "settings_for", counting)
    for _ in range(100):
        PiiRedactor().redact("line")
    assert len(asked) == 1


def test_a_pattern_that_does_not_compile_is_refused_at_startup(
    make_settings: MakeSettings,
) -> None:
    """Otherwise the first log line raises and redaction silently turns off."""
    with pytest.raises(ValueError, match="pii_extra_patterns"):
        make_settings(
            GuardrailSettingsMixin,
            guardrails_enabled=True,
            pii_extra_patterns={"bad": "("},
        )
