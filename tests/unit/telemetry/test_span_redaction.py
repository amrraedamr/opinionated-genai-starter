"""A prompt on a span, and whether it leaves the process.

`GENAI_PROMPT` and `GENAI_COMPLETION` carry whatever a user typed. A trace is
shipped to a collector somebody else runs and kept for weeks, so the copy on
the span matters as much as the copy the model saw.
"""

from collections.abc import Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

from base.services.configuration.mixins import GuardrailSettingsMixin
from base.services.telemetry.redaction import RedactingSpanExporter
from tests.conftest import MakeSettings

EMAIL = "bob@example.com"


class Capture(SpanExporter):
    """Stands in for the collector, and keeps what it was sent."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []
        self.flushed = 0
        self.shut = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.shut += 1

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        self.flushed += 1
        return True


def exported(attributes: dict[str, object]) -> Capture:
    """Run one span carrying `attributes` through the wrapper."""
    collector = Capture()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(collector)))
    with provider.get_tracer("test").start_as_current_span("call") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)  # type: ignore[arg-type]
    provider.force_flush()
    return collector


@pytest.fixture
def guarded(make_settings: MakeSettings) -> None:
    make_settings(GuardrailSettingsMixin, guardrails_enabled=True)


def test_a_prompt_does_not_reach_the_collector_with_an_address_in_it(
    guarded: None,
) -> None:
    collector = exported({"gen_ai.prompt": f"email me at {EMAIL}"})
    assert EMAIL not in str(collector.spans[0].attributes)


def test_what_the_attribute_was_about_still_survives(guarded: None) -> None:
    """A redacted trace is only useful if it still describes the call."""
    collector = exported({"gen_ai.prompt": f"email me at {EMAIL}"})
    assert "email me at" in str(collector.spans[0].attributes["gen_ai.prompt"])


def test_every_string_attribute_is_checked_not_just_the_known_ones(
    guarded: None,
) -> None:
    """An instrumentation library this service did not write also sets some."""
    collector = exported({"http.url": f"https://x/y?who={EMAIL}"})
    assert EMAIL not in str(collector.spans[0].attributes["http.url"])


def test_a_span_with_nothing_personal_on_it_is_passed_straight_through(
    guarded: None,
) -> None:
    """No copy when nothing changed, which is almost every span."""
    collector = exported({"http.method": "GET"})
    assert collector.spans[0].attributes["http.method"] == "GET"


def test_numbers_are_left_alone(guarded: None) -> None:
    """Token counts are not prose and scanning them protects nothing."""
    collector = exported({"gen_ai.usage.input_tokens": 42})
    assert collector.spans[0].attributes["gen_ai.usage.input_tokens"] == 42


def test_a_long_attribute_is_still_redacted(guarded: None) -> None:
    """A long prompt is exactly where an address hides."""
    collector = exported({"gen_ai.prompt": "x " * 20_000 + EMAIL})
    assert EMAIL not in str(collector.spans[0].attributes["gen_ai.prompt"])


def test_a_list_attribute_is_redacted(guarded: None) -> None:
    collector = exported({"gen_ai.messages": [f"to {EMAIL}", "hello"]})
    assert collector.spans[0].attributes["gen_ai.messages"] == (
        "to [redacted:email]",
        "hello",
    )


def test_an_event_and_an_exception_are_redacted(guarded: None) -> None:
    """`record_exception` copies the message, which often quotes the input."""
    collector = Capture()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(collector)))
    with provider.get_tracer("test").start_as_current_span("call") as span:
        span.add_event("note", {"body": f"from {EMAIL}"})
        span.record_exception(ValueError(f"bad user {EMAIL}"))
    provider.force_flush()

    events = collector.spans[0].events
    assert EMAIL not in str([dict(event.attributes or {}) for event in events])
    assert events[0].name == "note"


def test_a_status_description_is_redacted(guarded: None) -> None:
    collector = Capture()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(collector)))
    with provider.get_tracer("test").start_as_current_span("call") as span:
        span.set_status(Status(StatusCode.ERROR, f"failed for {EMAIL}"))
    provider.force_flush()

    status = collector.spans[0].status
    assert status.status_code == StatusCode.ERROR
    assert EMAIL not in str(status.description)


def test_a_redacted_span_still_reports_what_was_dropped(guarded: None) -> None:
    """The exporter sends dropped counts; a rebuilt span must not zero them."""
    collector = Capture()
    provider = TracerProvider(span_limits=SpanLimits(max_attributes=2))
    provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(collector)))
    with provider.get_tracer("test").start_as_current_span("call") as span:
        for index in range(4):
            span.set_attribute(f"extra.{index}", index)
        span.set_attribute("gen_ai.prompt", f"mail {EMAIL}")
    provider.force_flush()

    exported_span = collector.spans[0]
    assert EMAIL not in str(exported_span.attributes)
    assert exported_span.dropped_attributes == 3


def test_the_span_keeps_its_identity(guarded: None) -> None:
    """A rebuilt span that loses its context is a span nobody can find."""
    collector = exported({"gen_ai.prompt": f"mail {EMAIL}"})
    span = collector.spans[0]

    assert span.name == "call"
    assert span.context is not None
    assert span.start_time is not None
    assert span.end_time is not None


def test_nothing_is_redacted_until_a_service_asks(
    make_settings: MakeSettings,
) -> None:
    """It costs a pass over every attribute of every span."""
    make_settings(GuardrailSettingsMixin, guardrails_enabled=False)
    collector = exported({"gen_ai.prompt": f"mail {EMAIL}"})
    assert EMAIL in str(collector.spans[0].attributes["gen_ai.prompt"])


def test_shutdown_reaches_the_real_exporter() -> None:
    """A wrapper that swallows it leaves spans unflushed at exit."""
    collector = Capture()
    RedactingSpanExporter(collector).shutdown()
    assert collector.shut == 1


def test_a_flush_reaches_the_real_exporter() -> None:
    collector = Capture()
    assert RedactingSpanExporter(collector).force_flush() is True
    assert collector.flushed == 1
