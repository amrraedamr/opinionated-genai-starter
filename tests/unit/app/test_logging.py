"""Logging is where a secret leaks if nothing stops it, so redaction is the test.

The formatter also carries the correlation id onto every line, which is what
makes one request traceable across services after the fact.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from base.services.common.meta import Singleton
from base.services.logging.config import (
    JsonFormatter,
    RedactionFilter,
    logging_config,
    redact,
)
from base.services.logging.console import ConsoleLoggerService
from base.tools.correlation import adopt_correlation_id


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    Singleton.purge()
    yield
    Singleton.purge()


def record(message: str, **extra: Any) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
        **extra,
    )


@pytest.mark.parametrize(
    "line",
    [
        'password="hunter2"',
        "password=hunter2",
        "api_key: 'sk-live-1234'",
        "API-KEY=sk-live-1234",
        'token: "eyJhbGciOi"',
        "secret=shhh",
        "private_key=MIIEvQ",
    ],
)
def test_a_credential_is_redacted_whatever_it_is_called(line: str) -> None:
    assert "[REDACTED]" in redact(line)
    assert "hunter2" not in redact(line)


def test_an_authorization_header_is_redacted_bearer_and_all() -> None:
    assert redact("Authorization: Bearer eyJhbGciOi") == "Authorization: [REDACTED]"


def test_a_quoted_authorization_header_is_redacted_too() -> None:
    assert "[REDACTED]" in redact('{"authorization": "Basic abc123"}')
    assert "abc123" not in redact('{"authorization": "Basic abc123"}')


def test_ordinary_text_passes_through_untouched() -> None:
    assert redact("Ingested 12 chunks into handbook") == (
        "Ingested 12 chunks into handbook"
    )


def test_the_filter_rewrites_the_record_rather_than_only_the_output() -> None:
    entry = record("password=hunter2")

    assert RedactionFilter().filter(entry) is True
    assert "hunter2" not in entry.msg
    assert entry.args == ()


def test_the_filter_redacts_a_secret_hiding_in_a_format_argument() -> None:
    entry = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="connecting with %s",
        args=("password=hunter2",),
        exc_info=None,
    )

    RedactionFilter().filter(entry)

    assert "hunter2" not in entry.msg


def test_the_json_formatter_emits_one_parseable_object_per_line() -> None:
    correlation = adopt_correlation_id(str(uuid.uuid4()))

    payload = json.loads(JsonFormatter().format(record("hello")))

    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["correlation_id"] == str(correlation)
    assert payload["timestamp"]


def test_the_json_formatter_redacts_as_well() -> None:
    payload = json.loads(JsonFormatter().format(record("token=abc123")))

    assert "abc123" not in payload["message"]


def test_the_json_formatter_keeps_non_ascii_readable() -> None:
    payload = JsonFormatter().format(record("مرحبا"))

    assert "مرحبا" in payload


def test_an_exception_is_included_and_redacted() -> None:
    try:
        raise ValueError("password=hunter2")
    except ValueError:
        import sys

        entry = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    payload = json.loads(JsonFormatter().format(entry))

    assert "ValueError" in payload["exception"]
    assert "hunter2" not in payload["exception"]


def test_local_development_gets_colour_and_a_deployment_gets_json() -> None:
    assert (
        logging_config(environment="local")["handlers"]["console"]["formatter"]
        == "colored"
    )
    assert (
        logging_config(environment="production")["handlers"]["console"]["formatter"]
        == "json"
    )


def test_the_configured_level_reaches_both_the_handler_and_the_root() -> None:
    config = logging_config(level="DEBUG")

    assert config["handlers"]["console"]["level"] == "DEBUG"
    assert config["root"]["level"] == "DEBUG"


def test_redaction_is_always_attached_to_the_console_handler() -> None:
    assert logging_config()["handlers"]["console"]["filters"] == ["redaction"]


def test_existing_loggers_are_not_silenced_by_the_configuration() -> None:
    assert logging_config()["disable_existing_loggers"] is False


def test_the_console_service_is_one_instance_for_the_process() -> None:
    assert ConsoleLoggerService() is ConsoleLoggerService()


def test_an_error_line_carries_the_code_category_location_and_tag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ConsoleLoggerService().error(
        "it broke", status=503, location="RagRetrievalService", tag="Upstream"
    )

    out = capsys.readouterr().out
    assert "[Code: 503]" in out
    assert "[Location: RagRetrievalService]" in out
    assert "[Tag: Upstream]" in out
    assert "it broke" in out


def test_an_informational_line_is_left_as_written(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ConsoleLoggerService().info("plain message")

    out = capsys.readouterr().out
    assert "plain message" in out
    assert "[Code:" not in out


def test_a_secret_logged_by_accident_never_reaches_the_console(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ConsoleLoggerService().info("connecting with password=hunter2")

    out = capsys.readouterr().out
    assert "hunter2" not in out
    assert "[REDACTED]" in out


def test_the_logger_works_before_any_settings_are_registered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from base.services.configuration import is_configured

    assert is_configured() is False
    ConsoleLoggerService().info("still logs")

    assert "still logs" in capsys.readouterr().out
