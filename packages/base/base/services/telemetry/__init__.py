"""Tracing: one provider per worker, and the spans this repository opens.

Disabled unless a service turns it on, and silent when it is: `span()` hands
back a no-op where nothing is configured, so a caller writes the same code
either way and an unconfigured service pays an attribute lookup.

Correlation and tracing are one identity rather than two. A span adopts the
correlation id already in context, so a log line and a span can be put side by
side without a second id to carry.
"""

import importlib
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from base.models.definitions.telemetry import CORRELATION_ID, TENANT_ID
from base.services.common.meta import Singleton
from base.services.configuration import get_settings
from base.services.configuration.protocols import SupportsTelemetryConfig
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

    def configure(self, config: SupportsTelemetryConfig, *, app: Any = None) -> None:
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
                OTLPSpanExporter(endpoint=config.otel_exporter_endpoint, insecure=True)
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
        attributes: Mapping[str, Any] | None = None,
    ) -> Generator[Span | None]:
        """Open a span, or hand back None where tracing is off."""
        if self._provider is None:
            yield None
            return
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(name, kind=kind) as span:
            for key, value in self._ambient().items():
                span.set_attribute(key, value)
            for key, value in (attributes or {}).items():
                if value is not None:
                    span.set_attribute(key, value)
            yield span

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

    def _instrument(self, app: Any) -> None:
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
    def _instrument_fastapi(app: Any) -> None:
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
