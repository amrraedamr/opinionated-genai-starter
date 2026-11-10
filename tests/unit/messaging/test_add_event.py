"""Staging an event in the caller's transaction.

`add_event` is the producer half of the outbox pattern, and it is one call: it
builds the row and hands it to the session the caller already has, so the
event commits with the write or not at all. What it puts in that row is what a
dispatcher later reads, so the mapping is pinned here without a database.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from base.services.messaging import add_event
from contracts.events import Event


class Row:
    """The shape of a service's own outbox table, without the ORM."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class RecordingSession:
    """A session that keeps what was staged in it."""

    def __init__(self) -> None:
        self.added: list[Row] = []

    def add(self, row: Row) -> None:
        self.added.append(row)


def test_the_row_is_staged_in_the_session_it_was_given() -> None:
    session = RecordingSession()
    add_event(session, Row, Event.of("t", {}, producer="p"))  # type: ignore[arg-type]
    assert len(session.added) == 1


def test_the_row_id_is_the_event_id() -> None:
    session = RecordingSession()
    event = Event.of("t", {}, producer="p")
    add_event(session, Row, event)  # type: ignore[arg-type]
    assert session.added[0].id == event.meta.event_id


def test_the_row_carries_the_topic() -> None:
    session = RecordingSession()
    add_event(session, Row, Event.of("user.registered", {}, producer="p"))  # type: ignore[arg-type]
    assert session.added[0].topic == "user.registered"


def test_the_row_carries_the_event_as_json_ready_data() -> None:
    """A dispatcher stores this in a JSON column and validates it back."""
    session = RecordingSession()
    event = Event.of("t", {"a": 1}, producer="p", correlation_id=uuid4())
    add_event(session, Row, event)  # type: ignore[arg-type]
    stored = session.added[0].event
    assert isinstance(stored["meta"]["event_id"], str)
    assert Event.model_validate(stored) == event


def test_the_row_carries_when_the_event_occurred() -> None:
    session = RecordingSession()
    event = Event.of("t", {}, producer="p")
    add_event(session, Row, event)  # type: ignore[arg-type]
    assert session.added[0].occurred_at == event.meta.occurred_at
    assert session.added[0].occurred_at.tzinfo is not None


def test_nothing_else_is_filled_in() -> None:
    """`available_at`, `attempts` and `published_at` are the table's defaults."""
    session = RecordingSession()
    add_event(session, Row, Event.of("t", {}, producer="p"))  # type: ignore[arg-type]
    assert set(session.added[0].__dict__) == {"id", "topic", "event", "occurred_at"}


def test_two_events_stage_two_rows() -> None:
    session = RecordingSession()
    add_event(session, Row, Event.of("t", {}, producer="p"))  # type: ignore[arg-type]
    add_event(session, Row, Event.of("t", {}, producer="p"))  # type: ignore[arg-type]
    assert len({row.id for row in session.added}) == 2


def test_a_staged_event_round_trips_through_the_stored_payload() -> None:
    session = RecordingSession()
    event = Event.of(
        "t",
        {"nested": {"when": datetime(2026, 1, 1, tzinfo=UTC).isoformat()}, "u": "🎉"},
        producer="p",
    )
    add_event(session, Row, event)  # type: ignore[arg-type]
    restored = Event.model_validate(session.added[0].event)
    assert restored.payload == event.payload
    assert isinstance(restored.meta.event_id, UUID)


def test_the_tenant_in_effect_is_stamped_on_the_event() -> None:
    from base.models.definitions.tenancy import TenantId
    from base.tools.tenancy import TenantContext

    session = RecordingSession()
    with TenantContext.scope(TenantId("acme")):
        add_event(session, Row, Event.of("t", {}, producer="p"))  # type: ignore[arg-type]

    assert session.added[0].event["meta"]["tenant_id"] == "acme"


def test_an_event_produced_without_a_tenant_carries_none() -> None:
    session = RecordingSession()
    add_event(session, Row, Event.of("t", {}, producer="p"))  # type: ignore[arg-type]
    assert session.added[0].event["meta"]["tenant_id"] is None


def test_a_tenant_already_on_the_event_is_kept() -> None:
    from base.models.definitions.tenancy import TenantId
    from base.tools.tenancy import TenantContext

    session = RecordingSession()
    event = Event.of("t", {}, producer="p")
    event = event.model_copy(
        update={"meta": event.meta.model_copy(update={"tenant_id": "origin"})}
    )
    with TenantContext.scope(TenantId("other")):
        add_event(session, Row, event)  # type: ignore[arg-type]

    assert session.added[0].event["meta"]["tenant_id"] == "origin"
