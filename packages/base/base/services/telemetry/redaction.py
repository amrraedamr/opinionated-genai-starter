"""A span on its way out, with personal data taken off it first.

Spans are where prompts end up. `GENAI_PROMPT` and `GENAI_COMPLETION` carry
whatever a user typed, and a trace is routinely shipped to a collector run by
somebody else and kept for weeks. The guardrail that cleaned the prompt for
the model never saw the copy written onto the span.

This wraps the exporter rather than adding a processor, because by the time a
span ends its attributes are frozen: `BoundedAttributes` refuses mutation, so
`on_end` can read them and change nothing. Wrapping the last step instead
means every attribute is present and nothing has left the process yet.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from base.services.guardrails.sinks import MAX_ATTRIBUTE_LENGTH, redact_pii
from base.tools.errors import MissingExtraError

try:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "otel", "opentelemetry-sdk", feature="RedactingSpanExporter"
    ) from exc

if TYPE_CHECKING:
    from opentelemetry.util.types import AttributeValue


class RedactingSpanExporter(SpanExporter):
    """Passes spans on with their string attributes cleaned."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self._clean(span) for span in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)

    @staticmethod
    def _clean(span: ReadableSpan) -> ReadableSpan:
        """A copy of `span` with its attributes redacted, or the original.

        Rebuilt rather than edited because the attributes of a finished span
        are immutable. Returning the original when nothing changed keeps the
        common case free of a copy.
        """
        attributes = span.attributes
        if not attributes:
            return span

        cleaned: dict[str, AttributeValue] = {}
        changed = False
        for key, value in attributes.items():
            if isinstance(value, str) and len(value) <= MAX_ATTRIBUTE_LENGTH:
                redacted = redact_pii(value)
                changed = changed or redacted != value
                cleaned[key] = redacted
            else:
                cleaned[key] = value
        if not changed:
            return span

        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=cleaned,
            events=span.events,
            links=span.links,
            kind=span.kind,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )


__all__ = ["RedactingSpanExporter"]
