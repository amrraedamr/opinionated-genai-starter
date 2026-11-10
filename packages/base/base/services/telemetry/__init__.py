"""Tracing: one provider per worker, and the spans this repository opens."""

import importlib
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

from base.models.definitions.telemetry import CORRELATION_ID, TENANT_ID
from base.services.common.meta import Singleton
from base.services.configuration import get_settings
from base.services.configuration.protocols import SupportsTelemetryConfig
from base.services.telemetry.redaction import RedactingSpanExporter

if TYPE_CHECKING:
    from fastapi import FastAPI
    from opentelemetry.util.types import AttributeValue
from base.tools.correlation import correlation_id
from base.tools.errors import MissingExtraError
from base.tools.tenancy import TenantContext

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    from opentelemetry.trace import Span, SpanKind
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("otel", "opentelemetry-sdk", feature="Telemetry") from exc


class Telemetry(metaclass=Singleton):
    """The tracer this service exports with, and the spans it opens."""

    def __init__(self) -> None:
        self._provider: TracerProvider | None = None
        self._capture_content = False

    @property
    def enabled(self) -> bool:
        return self._provider is not None

    @property
    def capture_content(self) -> bool:
        """Whether prompts and completions may be recorded on a span."""
        return self._capture_content

    def configure(
        self, config: SupportsTelemetryConfig, *, app: "FastAPI | None" = None
    ) -> None:
        """Build the provider for this worker, and instrument what is installed."""
        self._capture_content = config.otel_capture_content
        if not config.otel_enabled or self._provider is not None:
            return

        service_name = config.otel_service_name or self._service_name()
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
            sampler=ParentBased(TraceIdRatioBased(config.otel_sample_ratio)),
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                RedactingSpanExporter(
                    OTLPSpanExporter(
                        endpoint=config.otel_exporter_endpoint, insecure=True
                    )
                )
            )
        )
        trace.set_tracer_provider(provider)
        self._provider = provider
        self._instrument(app)

    def shutdown(self) -> None:
        """Flush whatever is buffered, so a short-lived process still reports."""
        if self._provider is None:
            return
        self._provider.shutdown()
        self._provider = None

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: "Mapping[str, AttributeValue] | None" = None,
    ) -> Generator[Span | None]:
        """Open a span, or hand back None where tracing is off."""
        if self._provider is None:
            yield None
            return
        # Own provider, not the global one, which another library may have claimed
        tracer = self._provider.get_tracer(__name__)
        with tracer.start_as_current_span(name, kind=kind) as span:
            for key, value in self._ambient().items():
                span.set_attribute(key, value)
            for key, value in (attributes or {}).items():
                if value is not None:
                    span.set_attribute(key, value)
            yield span

    @contextmanager
    def continued_span(
        self,
        name: str,
        traceparent: str | None,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: "Mapping[str, AttributeValue] | None" = None,
    ) -> Generator[Span | None]:
        """Open a span continuing `traceparent`, or hand back None where tracing is off.

        An absent traceparent continues the current context; an unparseable one starts a fresh trace.
        """
        if self._provider is None:
            yield None
            return
        parent = (
            TraceContextTextMapPropagator().extract({"traceparent": traceparent})
            if traceparent
            else None
        )
        # Own provider, not the global one, which another library may have claimed
        tracer = self._provider.get_tracer(__name__)
        with tracer.start_as_current_span(name, context=parent, kind=kind) as span:
            for key, value in self._ambient().items():
                span.set_attribute(key, value)
            for key, value in (attributes or {}).items():
                if value is not None:
                    span.set_attribute(key, value)
            yield span

    @staticmethod
    def current_traceparent() -> str | None:
        """This span, in the form another process can continue it from; `None` when untraced."""
        carrier: dict[str, str] = {}
        TraceContextTextMapPropagator().inject(carrier)
        return carrier.get("traceparent")

    @staticmethod
    def _ambient() -> dict[str, str]:
        """What every span in this process should carry, where it is known."""
        ambient: dict[str, str] = {}
        current = correlation_id.get(None)
        if current is not None:
            ambient[CORRELATION_ID] = str(current)
        tenant = TenantContext.get()
        if tenant is not None:
            ambient[TENANT_ID] = tenant
        return ambient

    @staticmethod
    def _service_name() -> str:
        return get_settings().service_name

    def _instrument(self, app: "FastAPI | None") -> None:
        """Instrument the libraries this build installed, and skip the rest."""
        if app is not None:
            self._instrument_fastapi(app)
        self._instrument_module(
            "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"
        )
        self._instrument_module(
            "opentelemetry.instrumentation.redis", "RedisInstrumentor"
        )
        self._instrument_module(
            "opentelemetry.instrumentation.sqlalchemy", "SQLAlchemyInstrumentor"
        )

    @staticmethod
    def _instrument_fastapi(app: "FastAPI") -> None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        except ModuleNotFoundError:  # pragma: no cover
            return
        FastAPIInstrumentor.instrument_app(app)

    @staticmethod
    def _instrument_module(module_name: str, attribute: str) -> None:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:  # pragma: no cover
            return
        getattr(module, attribute)().instrument()


__all__ = ["Telemetry"]
