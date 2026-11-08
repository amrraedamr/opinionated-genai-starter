"""Tracing: off unless asked for, and silent about content unless asked twice.

The properties that matter are the negative ones. A service that did not turn
tracing on must pay nothing and export nothing, and a service that did must
still not put prompts or completions on a span unless it separately said so —
a trace backend is a third party, and a prompt is user data.

Nothing here reaches a collector: spans are captured in memory, which is what
makes the assertions exact rather than a check that something was sent.
"""

import contextvars
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from base.models.definitions.telemetry import (
    CORRELATION_ID,
    GENAI_COMPLETION,
    GENAI_PROMPT,
    GENAI_REQUEST_MODEL,
    TENANT_ID,
)
from base.models.definitions.tenancy import TenantId
from base.services.configuration.mixins import TelemetrySettingsMixin
from base.services.telemetry import Telemetry
from base.tools.correlation import correlation_id
from base.tools.tenancy import TenantContext


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    """A configured telemetry singleton writing spans into memory."""
    captured = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(captured))

    service = Telemetry()
    service._provider = provider  # type: ignore[assignment]
    previous = trace._TRACER_PROVIDER  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    try:
        yield captured
    finally:
        trace._TRACER_PROVIDER = previous  # type: ignore[attr-defined]
        service._provider = None  # type: ignore[assignment]


def settings(**values: Any) -> TelemetrySettingsMixin:
    return TelemetrySettingsMixin(**values)


# Off by default


def test_tracing_is_off_until_a_service_turns_it_on() -> None:
    assert Telemetry().enabled is False


def test_configuring_with_it_disabled_builds_no_provider() -> None:
    service = Telemetry()
    service.configure(settings(otel_enabled=False))
    assert service.enabled is False


def test_a_span_opened_while_disabled_is_handed_back_as_nothing() -> None:
    with Telemetry().span("anything") as span:
        assert span is None


def test_a_disabled_shutdown_is_a_no_op() -> None:
    Telemetry().shutdown()
    assert Telemetry().enabled is False


def test_the_default_endpoint_is_the_otlp_grpc_one() -> None:
    assert settings().otel_exporter_endpoint == "http://localhost:4317"


def test_everything_is_sampled_unless_a_ratio_says_otherwise() -> None:
    assert settings().otel_sample_ratio == 1.0


# Spans


def test_a_span_is_recorded_with_the_name_it_was_given(
    exporter: InMemorySpanExporter,
) -> None:
    with Telemetry().span("ask-the-model"):
        pass
    assert [s.name for s in exporter.get_finished_spans()] == ["ask-the-model"]


def test_a_span_carries_the_attributes_it_was_given(
    exporter: InMemorySpanExporter,
) -> None:
    with Telemetry().span("call", attributes={GENAI_REQUEST_MODEL: "gpt-5"}):
        pass
    span = exporter.get_finished_spans()[0]
    assert span.attributes[GENAI_REQUEST_MODEL] == "gpt-5"


def test_an_attribute_with_no_value_is_left_off_rather_than_set_to_none(
    exporter: InMemorySpanExporter,
) -> None:
    with Telemetry().span("call", attributes={GENAI_REQUEST_MODEL: None}):
        pass
    assert GENAI_REQUEST_MODEL not in exporter.get_finished_spans()[0].attributes


def test_a_span_records_the_kind_it_was_opened_as(
    exporter: InMemorySpanExporter,
) -> None:
    with Telemetry().span("publish", kind=SpanKind.PRODUCER):
        pass
    assert exporter.get_finished_spans()[0].kind == SpanKind.PRODUCER


def test_spans_opened_inside_one_another_share_a_trace(
    exporter: InMemorySpanExporter,
) -> None:
    service = Telemetry()
    with service.span("outer"):
        with service.span("inner"):
            pass
    finished = {s.name: s for s in exporter.get_finished_spans()}
    inner, outer = finished["inner"], finished["outer"]
    assert inner.context.trace_id == outer.context.trace_id
    assert inner.parent.span_id == outer.context.span_id


def test_an_exception_inside_a_span_still_ends_it(
    exporter: InMemorySpanExporter,
) -> None:
    with pytest.raises(RuntimeError):
        with Telemetry().span("doomed"):
            raise RuntimeError("boom")
    assert len(exporter.get_finished_spans()) == 1


# Correlation and tenancy ride along


def test_a_span_adopts_the_correlation_id_already_in_context(
    exporter: InMemorySpanExporter,
) -> None:
    """One identity, not two: a log line and a span can be put side by side."""
    known = uuid.uuid4()
    token = correlation_id.set(known)
    try:
        with Telemetry().span("work"):
            pass
    finally:
        correlation_id.reset(token)
    assert exporter.get_finished_spans()[0].attributes[CORRELATION_ID] == str(known)


def test_a_span_without_a_correlation_id_simply_does_not_carry_one(
    exporter: InMemorySpanExporter,
) -> None:
    """Run in a fresh context: the id is a `ContextVar` an earlier caller may hold."""

    def work() -> None:
        with Telemetry().span("work"):
            pass

    contextvars.Context().run(work)
    assert CORRELATION_ID not in exporter.get_finished_spans()[0].attributes


def test_a_span_carries_the_tenant_being_acted_for(
    exporter: InMemorySpanExporter,
) -> None:
    with TenantContext.scope(TenantId("acme")):
        with Telemetry().span("work"):
            pass
    assert exporter.get_finished_spans()[0].attributes[TENANT_ID] == "acme"


def test_a_span_outside_any_tenant_does_not_carry_one(
    exporter: InMemorySpanExporter,
) -> None:
    with Telemetry().span("work"):
        pass
    assert TENANT_ID not in exporter.get_finished_spans()[0].attributes


# Content


def test_prompt_content_is_not_recorded_unless_a_service_asks_for_it() -> None:
    service = Telemetry()
    service.configure(settings(otel_enabled=False))
    assert service.capture_content is False


def test_content_capture_is_opt_in_separately_from_tracing_itself() -> None:
    service = Telemetry()
    service.configure(settings(otel_enabled=False, otel_capture_content=True))
    assert service.capture_content is True


def test_the_content_attributes_are_named_but_never_set_by_the_seam(
    exporter: InMemorySpanExporter,
) -> None:
    """The names exist for adapters that opt in; the seam itself sets neither."""
    with Telemetry().span("call"):
        pass
    attributes = exporter.get_finished_spans()[0].attributes
    assert GENAI_PROMPT not in attributes
    assert GENAI_COMPLETION not in attributes


# Settings


def test_the_mixin_defaults_leave_a_service_exporting_nothing() -> None:
    config = settings()
    assert (config.otel_enabled, config.otel_capture_content) == (False, False)


def test_the_service_name_falls_back_to_the_services_own_name() -> None:
    assert settings().otel_service_name == ""
