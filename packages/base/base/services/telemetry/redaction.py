"""Redacts at the exporter: a finished span's attributes are immutable."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from base.services.guardrails.sinks import PiiRedactor
from base.tools.errors import MissingExtraError

try:
    from opentelemetry.attributes import BoundedAttributes
    from opentelemetry.sdk.trace import Event, ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.sdk.util import BoundedList
    from opentelemetry.trace import Link, Status
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "otel", "opentelemetry-sdk", feature="RedactingSpanExporter"
    ) from exc

if TYPE_CHECKING:
    from opentelemetry.util.types import AnyValue


class RedactingSpanExporter(SpanExporter):
    """Passes spans on with personal data removed from every string they carry."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner
        self._redactor = PiiRedactor()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self._clean(span) for span in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def _clean(self, span: ReadableSpan) -> ReadableSpan:
        """A redacted copy of `span`, or `span` itself when nothing changed."""
        attributes = self._attributes(span.attributes)
        events = [self._event(event) for event in span.events]
        links = [self._link(link) for link in span.links]
        status = self._status(span.status)
        unchanged = (
            attributes is None
            and all(new is old for new, old in zip(events, span.events, strict=True))
            and all(new is old for new, old in zip(links, span.links, strict=True))
            and status is span.status
        )
        if unchanged:
            return span

        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=self._bounded_attributes(
                attributes or span.attributes, span.dropped_attributes
            ),
            events=self._bounded(events, span.dropped_events),
            links=self._bounded(links, span.dropped_links),
            kind=span.kind,
            status=status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def _attributes(
        self, attributes: Mapping[str, "AnyValue"] | None
    ) -> dict[str, "AnyValue"] | None:
        """Redacted attributes, or None when nothing in them changed."""
        if not attributes:
            return None
        cleaned = {key: self._value(value) for key, value in attributes.items()}
        if all(cleaned[key] is value for key, value in attributes.items()):
            return None
        return cleaned

    def _value(self, value: "AnyValue") -> "AnyValue":
        """A string, or a sequence of them, redacted; anything else as given."""
        if isinstance(value, str):
            redacted = self._redactor.redact(value)
            return value if redacted == value else redacted
        if isinstance(value, Sequence):
            strings = [item for item in value if isinstance(item, str)]
            if strings and len(strings) == len(value):
                redacted_items = tuple(self._redactor.redact(item) for item in strings)
                return value if redacted_items == tuple(strings) else redacted_items
        return value

    def _event(self, event: Event) -> Event:
        attributes = self._attributes(event.attributes)
        if attributes is None:
            return event
        bounded = self._bounded_attributes(attributes, event.dropped_attributes)
        return Event(event.name, bounded, event.timestamp)

    def _link(self, link: Link) -> Link:
        attributes = self._attributes(link.attributes)
        if attributes is None:
            return link
        return Link(
            link.context, self._bounded_attributes(attributes, link.dropped_attributes)
        )

    def _status(self, status: Status) -> Status:
        if not status.description:
            return status
        redacted = self._redactor.redact(status.description)
        if redacted == status.description:
            return status
        return Status(status.status_code, redacted)

    @staticmethod
    def _bounded_attributes(
        attributes: Mapping[str, "AnyValue"] | None, dropped: int
    ) -> BoundedAttributes:
        """Attributes carrying the dropped count the exporter reports."""
        bounded = BoundedAttributes(attributes=attributes)
        bounded.dropped = dropped
        return bounded

    @staticmethod
    def _bounded[T](items: list[T], dropped: int) -> Sequence[T]:
        """Items carrying the dropped count the exporter reports, when there was one."""
        if not dropped:
            return items
        bounded = BoundedList.from_seq(None, items)
        bounded.dropped = dropped
        return bounded


__all__ = ["RedactingSpanExporter"]
