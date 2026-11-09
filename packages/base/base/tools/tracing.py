"""Spans for code that has to run whether or not tracing is installed.

`base.services.telemetry` raises on import without the `otel` extra, which is
right for a service that asked for tracing and did not get it. It is wrong for
the outbox and the inbox: those are core machinery, and the worker that runs
them does not carry the extra. Importing them must not depend on it.

This cannot live under `base.services.telemetry`, because importing any
submodule of a package runs that package's `__init__` first — which is the
module that raises.
"""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

from base.tools.errors import MissingExtraError

if TYPE_CHECKING:
    from opentelemetry.trace import Span
    from opentelemetry.util.types import AttributeValue

try:
    from base.services.telemetry import Telemetry

    _AVAILABLE = True
except MissingExtraError:  # pragma: no cover - depends on the installed extras
    _AVAILABLE = False


class OptionalTelemetry:
    """The two span shapes messaging needs, and nothing when tracing is absent."""

    AVAILABLE: ClassVar[bool] = _AVAILABLE

    @staticmethod
    @contextmanager
    def continued_span(
        name: str,
        traceparent: str | None,
        *,
        kind: str = "internal",
        attributes: "Mapping[str, AttributeValue] | None" = None,
    ) -> "Generator[Span | None]":
        """A span continuing `traceparent`, or nothing at all."""
        if not _AVAILABLE:
            yield None
            return
        from opentelemetry.trace import SpanKind

        kinds = {
            "producer": SpanKind.PRODUCER,
            "consumer": SpanKind.CONSUMER,
            "client": SpanKind.CLIENT,
            "server": SpanKind.SERVER,
            "internal": SpanKind.INTERNAL,
        }
        with Telemetry().continued_span(
            name, traceparent, kind=kinds[kind], attributes=attributes
        ) as span:
            yield span

    @staticmethod
    def current_traceparent() -> str | None:
        """This span, for another process to continue from. `None` when untraced."""
        if not _AVAILABLE:
            return None
        return Telemetry().current_traceparent()

    @staticmethod
    def capture_content() -> bool:
        """Whether this service opted into prompts and replies on its spans."""
        if not _AVAILABLE:
            return False
        return Telemetry().capture_content


__all__ = ["OptionalTelemetry"]
