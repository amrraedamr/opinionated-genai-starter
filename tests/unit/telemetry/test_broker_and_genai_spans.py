"""Traces that survive a process boundary, and spans that do not leak prompts.

Two questions only a real exporter can answer. Does a consumer's work join the
trace that produced the event, rather than starting an orphan? And does the
prompt stay out of the span unless a service asked for it — a trace is
routinely shipped to a third party, and the prompt is the one thing in it that
belongs to somebody else.

An in-memory exporter is used rather than the collector: this is about what
the spans contain, not whether they travel, which `tests/integration` covers.
"""

from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind

from base.models.definitions.telemetry import (
    GENAI_COMPLETION,
    GENAI_OPERATION,
    GENAI_PROMPT,
    GENAI_REQUEST_MODEL,
    GENAI_SYSTEM,
    GENAI_USAGE_INPUT_TOKENS,
    GENAI_USAGE_OUTPUT_TOKENS,
    MESSAGING_DESTINATION,
    TENANT_ID,
)
from base.models.definitions.tenancy import TenantId
from base.models.view_models.llm_result import LLMResult, LLMUsageCounts
from base.services.configuration.mixins import (
    RateLimitSettingsMixin,
    TelemetrySettingsMixin,
    TenancySettingsMixin,
)
from base.services.llm import AbstractLLMService
from base.services.telemetry import Telemetry
from base.tools.tenancy import TenantContext
from base.tools.tracing import OptionalTelemetry
from contracts.events import Event


class ScriptedLLM(AbstractLLMService):
    """A provider that answers immediately, so the span is the subject."""

    @property
    def model(self) -> str:
        return "test-model"

    def get_service_name(self) -> str:
        return "scripted"

    def _apply_cache_hint(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def _complete(self, *args: Any, **kwargs: Any) -> LLMResult:
        return LLMResult(
            content="the reply",
            usage=LLMUsageCounts(input_tokens=11, output_tokens=7),
        )

    async def _acomplete(self, *args: Any, **kwargs: Any) -> LLMResult:
        return self._complete()

    def _parse(self, schema: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def _aparse(self, schema: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def exporter(make_settings: Any) -> InMemorySpanExporter:
    """A provider whose spans stay in this process, installed on Telemetry."""

    def _build(*, capture_content: bool = False, **extra: Any) -> InMemorySpanExporter:
        make_settings(
            TelemetrySettingsMixin,
            RateLimitSettingsMixin,
            TenancySettingsMixin,
            service_name="span-test",
            otel_enabled=True,
            otel_capture_content=capture_content,
            **extra,
        )
        memory = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(memory))
        telemetry = Telemetry()
        telemetry._provider = provider
        telemetry._capture_content = capture_content
        return memory

    return _build  # type: ignore[return-value]


def named(exporter: InMemorySpanExporter, fragment: str) -> list[Any]:
    return [s for s in exporter.get_finished_spans() if fragment in s.name]


def test_a_consumer_joins_the_trace_that_produced_the_event(
    exporter: Any,
) -> None:
    """Without this each side is its own orphan and the link is a timestamp."""
    exporter()
    telemetry = Telemetry()

    with telemetry.span("producing") as producer:
        assert producer is not None
        traceparent = Telemetry.current_traceparent()
        produced_trace = producer.get_span_context().trace_id

    assert traceparent is not None
    with telemetry.continued_span("inbox process orders", traceparent) as consumer:
        assert consumer is not None
        assert consumer.get_span_context().trace_id == produced_trace


def test_a_missing_traceparent_starts_a_fresh_trace(exporter: Any) -> None:
    """A published event is not worth losing over an absent header."""
    memory = exporter()
    telemetry = Telemetry()
    with telemetry.continued_span("inbox process orders", None) as span:
        assert span is not None
    assert named(memory, "inbox process orders")


def test_an_unparseable_traceparent_does_not_raise(exporter: Any) -> None:
    memory = exporter()
    telemetry = Telemetry()
    with telemetry.continued_span("inbox process orders", "not-a-traceparent"):
        pass
    assert named(memory, "inbox process orders")


def test_an_event_carries_the_traceparent_field() -> None:
    """The contract has somewhere to put it, defaulting to absent."""
    event = Event.of("orders.created", {"id": 1}, producer="test")
    assert event.meta.traceparent is None
    revived = Event.model_validate(event.model_dump(mode="json"))
    assert revived.meta.traceparent is None


def test_the_traceparent_survives_serialisation() -> None:
    """It travels as JSON through the outbox table, so it must round trip."""
    event = Event.of("orders.created", {"id": 1}, producer="test")
    carried = event.model_copy(
        update={"meta": event.meta.model_copy(update={"traceparent": "00-abc-def-01"})}
    )
    revived = Event.model_validate(carried.model_dump(mode="json"))
    assert revived.meta.traceparent == "00-abc-def-01"


def test_a_model_call_records_what_it_cost(exporter: Any) -> None:
    """Tokens on the span are how a trace answers "what did this turn cost"."""
    memory = exporter()
    from base.models.view_models.prompt import RenderedPrompt

    ScriptedLLM().complete(prompt=RenderedPrompt(messages=[]))

    spans = named(memory, "chat test-model")
    assert spans, "no GenAI span was emitted"
    attributes = spans[0].attributes
    assert attributes[GENAI_SYSTEM] == "scripted"
    assert attributes[GENAI_OPERATION] == "chat"
    assert attributes[GENAI_REQUEST_MODEL] == "test-model"
    assert attributes[GENAI_USAGE_INPUT_TOKENS] == 11
    assert attributes[GENAI_USAGE_OUTPUT_TOKENS] == 7


def test_the_prompt_is_not_an_attribute_by_default(exporter: Any) -> None:
    """A trace goes to a third party; the prompt is somebody else's."""
    memory = exporter(capture_content=False)
    from base.models.view_models.prompt import RenderedPrompt

    ScriptedLLM().complete(prompt=RenderedPrompt(messages=[]))

    attributes = named(memory, "chat test-model")[0].attributes
    assert GENAI_PROMPT not in attributes
    assert GENAI_COMPLETION not in attributes


def test_the_prompt_is_an_attribute_when_a_service_asks(exporter: Any) -> None:
    memory = exporter(capture_content=True)
    from base.models.definitions.message import MessageRole
    from base.models.view_models.prompt import (
        RenderedPrompt,
        RenderedPromptMessage,
    )

    prompt = RenderedPrompt(
        messages=[RenderedPromptMessage(role=MessageRole.USER, content="hello there")]
    )
    ScriptedLLM().complete(prompt=prompt)

    attributes = named(memory, "chat test-model")[0].attributes
    assert "hello there" in attributes[GENAI_PROMPT]
    assert attributes[GENAI_COMPLETION] == "the reply"


def test_a_genai_span_is_a_client_span(exporter: Any) -> None:
    """It describes a call leaving this process, not work done inside it."""
    memory = exporter()
    from base.models.view_models.prompt import RenderedPrompt

    ScriptedLLM().complete(prompt=RenderedPrompt(messages=[]))
    assert named(memory, "chat test-model")[0].kind is SpanKind.CLIENT


def test_a_span_carries_the_tenant_when_tenancy_is_on(exporter: Any) -> None:
    """So a trace can be read per customer rather than only in aggregate."""
    memory = exporter(tenancy_enabled=True)
    telemetry = Telemetry()
    with TenantContext.scope(TenantId("acme")):
        with telemetry.span("work"):
            pass
    assert named(memory, "work")[0].attributes[TENANT_ID] == "acme"


def test_a_messaging_span_names_its_topic(exporter: Any) -> None:
    memory = exporter()
    with OptionalTelemetry.continued_span(
        "outbox dispatch orders.created",
        None,
        kind="producer",
        attributes={MESSAGING_DESTINATION: "orders.created"},
    ):
        pass
    span = named(memory, "outbox dispatch")[0]
    assert span.attributes[MESSAGING_DESTINATION] == "orders.created"
    assert span.kind is SpanKind.PRODUCER


def test_tracing_is_optional_for_code_that_cannot_require_it() -> None:
    """The worker runs the outbox without the otel extra; it must not break."""
    assert isinstance(OptionalTelemetry.AVAILABLE, bool)
    with OptionalTelemetry.continued_span("anything", None) as span:
        assert span is None or span is not None
