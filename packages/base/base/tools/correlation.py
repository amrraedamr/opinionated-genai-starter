"""Correlation identity for one logical request across services."""

import contextvars
import re
from uuid import UUID, uuid4

from base.tools.misc import parse_uuid

CORRELATION_HEADER = "X-Correlation-ID"
TRACEPARENT_HEADER = "traceparent"

_TRACEPARENT = re.compile(r"^\d{2}-([0-9a-f]{32})-[0-9a-f]{16}-\d{2}$")

correlation_id: contextvars.ContextVar[UUID] = contextvars.ContextVar("correlation_id")


def parse_correlation_id(value: str | None) -> UUID | None:
    """Read a correlation id from a header value, or return None."""
    if not value:
        return None
    resolved = parse_uuid(value)
    if resolved is not None:
        return resolved
    match = _TRACEPARENT.match(value.strip())
    return UUID(match.group(1)) if match else None


def adopt_correlation_id(
    header_value: str | None = None, traceparent: str | None = None
) -> UUID:
    """Adopt an inbound correlation id, or start a new one."""
    resolved = (
        parse_correlation_id(header_value)
        or parse_correlation_id(traceparent)
        or uuid4()
    )
    correlation_id.set(resolved)
    return resolved


def current_correlation_id() -> UUID:
    """The current correlation id, starting one if the context has none."""
    try:
        return correlation_id.get()
    except LookupError:
        return adopt_correlation_id()
