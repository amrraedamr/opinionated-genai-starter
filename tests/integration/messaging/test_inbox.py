"""The inbox against the real database.

Delivery is at-least-once, so the same event arrives more than once and the
inbox is what stops the handler running twice. Everything here is about that
claim: who gets it, who is turned away, and what happens to a claim whose
process died holding it.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from base.models.definitions.messaging import InboxState
from base.models.view_models.exceptions.sql import SQLConflictError
from base.services.messaging import InboxProcessor
from base.services.sql.service import SQLService
from contracts.events import Event
from tests.integration.messaging.conftest import InboxEventRow, event

pytestmark = pytest.mark.integration


class Recorder:
    """A handler that counts its runs and can be told to fail."""

    def __init__(self, failing: bool = False) -> None:
        self.runs: list[Event] = []
        self.failing = failing

    async def __call__(self, incoming: Event) -> None:
        self.runs.append(incoming)
        if self.failing:
            raise ValueError("handler failed")

    @property
    def count(self) -> int:
        return len(self.runs)


async def stored(service: SQLService, delivered: Event) -> InboxEventRow:
    async with service.atransaction() as session:
        row = await session.get(InboxEventRow, delivered.meta.event_id)
        assert row is not None
        return row


async def row_count(service: SQLService) -> int:
    async with service.atransaction() as session:
        return int(
            await session.scalar(select(func.count()).select_from(InboxEventRow)) or 0
        )


async def set_claim(
    service: SQLService, delivered: Event, *, state: InboxState, age_seconds: float
) -> None:
    """Rewrite a claim, so a test need not wait out the claim timeout."""
    async with service.atransaction() as session:
        row = await session.get(InboxEventRow, delivered.meta.event_id)
        assert row is not None
        row.state = state
        row.locked_at = datetime.now(UTC) - timedelta(seconds=age_seconds)


async def test_a_first_delivery_runs_the_handler(messaging_tables: Any) -> None:
    del messaging_tables
    handler = Recorder()
    delivered = event()
    await InboxProcessor(InboxEventRow).wrap(handler)(delivered)
    assert [e.meta.event_id for e in handler.runs] == [delivered.meta.event_id]


async def test_the_same_event_delivered_twice_runs_the_handler_once(
    messaging_tables: Any,
) -> None:
    del messaging_tables
    handler = Recorder()
    wrapped = InboxProcessor(InboxEventRow).wrap(handler)
    delivered = event()
    await wrapped(delivered)
    await wrapped(delivered)
    assert handler.count == 1


async def test_a_redelivery_through_a_new_processor_still_runs_it_once(
    messaging_tables: Any,
) -> None:
    """A restart must not replay what the previous process already handled."""
    del messaging_tables
    handler = Recorder()
    delivered = event()
    await InboxProcessor(InboxEventRow).wrap(handler)(delivered)
    await InboxProcessor(InboxEventRow).wrap(handler)(delivered)
    assert handler.count == 1


async def test_two_different_events_both_run(messaging_tables: Any) -> None:
    del messaging_tables
    handler = Recorder()
    wrapped = InboxProcessor(InboxEventRow).wrap(handler)
    await wrapped(event(n=1))
    await wrapped(event(n=2))
    assert handler.count == 2


async def test_a_handled_event_is_recorded_as_processed(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    delivered = event("user.registered")
    await InboxProcessor(InboxEventRow).wrap(Recorder())(delivered)
    row = await stored(service, delivered)
    assert row.state == InboxState.PROCESSED
    assert row.topic == "user.registered"
    assert row.processed_at is not None
    assert row.last_error is None
    assert row.attempts == 1


async def test_a_failing_handler_is_recorded_as_failed_and_raises(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    delivered = event()
    with pytest.raises(ValueError, match="handler failed"):
        await InboxProcessor(InboxEventRow).wrap(Recorder(failing=True))(delivered)
    row = await stored(service, delivered)
    assert row.state == InboxState.FAILED
    assert row.processed_at is None
    assert row.last_error == "ValueError: handler failed"


async def test_a_failed_event_is_claimed_again_on_redelivery(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    failing, succeeding = Recorder(failing=True), Recorder()
    processor = InboxProcessor(InboxEventRow)
    delivered = event()
    with pytest.raises(ValueError):
        await processor.wrap(failing)(delivered)
    await processor.wrap(succeeding)(delivered)
    row = await stored(service, delivered)
    assert succeeding.count == 1
    assert row.state == InboxState.PROCESSED
    assert row.attempts == 2
    assert row.last_error is None


async def test_a_claim_still_in_flight_turns_a_duplicate_away(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    handler = Recorder()
    processor = InboxProcessor(InboxEventRow)
    delivered = event()
    await processor.wrap(handler)(delivered)
    await set_claim(service, delivered, state=InboxState.PROCESSING, age_seconds=1)
    await processor.wrap(handler)(delivered)
    assert handler.count == 1


async def test_a_claim_older_than_the_timeout_is_taken_over(
    messaging_tables: Any,
) -> None:
    """The process holding it died, so somebody has to finish the work."""
    service = messaging_tables
    handler = Recorder()
    processor = InboxProcessor(InboxEventRow, claim_timeout_seconds=30)
    delivered = event()
    await processor.wrap(handler)(delivered)
    await set_claim(service, delivered, state=InboxState.PROCESSING, age_seconds=600)
    await processor.wrap(handler)(delivered)
    assert handler.count == 2
    assert (await stored(service, delivered)).state == InboxState.PROCESSED


async def test_the_claim_timeout_is_honoured(messaging_tables: Any) -> None:
    service = messaging_tables
    handler = Recorder()
    processor = InboxProcessor(InboxEventRow, claim_timeout_seconds=3_600)
    delivered = event()
    await processor.wrap(handler)(delivered)
    await set_claim(service, delivered, state=InboxState.PROCESSING, age_seconds=600)
    await processor.wrap(handler)(delivered)
    assert handler.count == 1


async def test_taking_over_a_stale_claim_counts_an_attempt(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    processor = InboxProcessor(InboxEventRow, claim_timeout_seconds=30)
    delivered = event()
    await processor.wrap(Recorder())(delivered)
    for _ in range(3):
        await set_claim(
            service, delivered, state=InboxState.PROCESSING, age_seconds=600
        )
        await processor.wrap(Recorder())(delivered)
    assert (await stored(service, delivered)).attempts == 4


async def test_concurrent_duplicates_run_the_handler_once(
    messaging_tables: Any,
) -> None:
    """At-least-once delivery means two consumers can race on one event.

    Exactly one claim wins. The losers see the row that beat them and return,
    or lose the insert and raise a conflict, but the handler never runs twice
    and the table never holds two rows for one event.
    """
    service = messaging_tables
    handler = Recorder()
    wrapped = InboxProcessor(InboxEventRow).wrap(handler)
    delivered = event()

    outcomes = await asyncio.gather(
        *(wrapped(delivered) for _ in range(4)), return_exceptions=True
    )
    raised = [o for o in outcomes if isinstance(o, BaseException)]

    assert handler.count == 1
    assert await row_count(service) == 1
    assert all(isinstance(error, SQLConflictError) for error in raised)
    assert len(raised) <= 3


async def test_concurrent_distinct_events_all_run(messaging_tables: Any) -> None:
    del messaging_tables
    handler = Recorder()
    wrapped = InboxProcessor(InboxEventRow).wrap(handler)
    events = [event(n=n) for n in range(6)]

    await asyncio.gather(*(wrapped(delivered) for delivered in events))
    assert {e.meta.event_id for e in handler.runs} == {e.meta.event_id for e in events}


async def test_a_claim_that_disappears_is_reported(messaging_tables: Any) -> None:
    """Another process truncating the table mid-handler must not pass silently."""
    service = messaging_tables
    processor = InboxProcessor(InboxEventRow)
    delivered = event()

    async def handler(incoming: Event) -> None:
        async with service.atransaction() as session:
            row = await session.get(InboxEventRow, incoming.meta.event_id)
            await session.delete(row)

    with pytest.raises(RuntimeError, match="Inbox claim disappeared"):
        await processor.wrap(handler)(delivered)


async def test_a_slow_handler_keeps_its_claim(messaging_tables: Any) -> None:
    service = messaging_tables
    processor = InboxProcessor(InboxEventRow)
    delivered = event()
    states: list[str] = []

    async def handler(incoming: Event) -> None:
        await asyncio.sleep(0.2)
        states.append((await stored(service, incoming)).state)

    await processor.wrap(handler)(delivered)
    assert states == [InboxState.PROCESSING]
    assert (await stored(service, delivered)).state == InboxState.PROCESSED


async def test_an_event_with_an_empty_payload_is_handled(
    messaging_tables: Any,
) -> None:
    del messaging_tables
    handler = Recorder()
    delivered = Event.of("test.messaging", {}, producer="test-producer")
    await InboxProcessor(InboxEventRow).wrap(handler)(delivered)
    assert handler.runs[0].payload == {}


async def test_a_large_payload_is_handled(messaging_tables: Any) -> None:
    del messaging_tables
    handler = Recorder()
    payload = {"blob": "x" * 200_000}
    delivered = Event.of("test.messaging", payload, producer="test-producer")
    await InboxProcessor(InboxEventRow).wrap(handler)(delivered)
    assert handler.runs[0].payload == payload


async def test_a_long_topic_is_stored(messaging_tables: Any) -> None:
    """The column is 255 characters, so a longer topic would be rejected."""
    service = messaging_tables
    topic = "t." + "x" * 250
    delivered = Event.of(topic, {}, producer="test-producer")
    await InboxProcessor(InboxEventRow).wrap(Recorder())(delivered)
    assert (await stored(service, delivered)).topic == topic


async def test_a_unicode_topic_is_stored(messaging_tables: Any) -> None:
    service = messaging_tables
    topic = "tópico.ünicode.🎉"
    delivered = Event.of(topic, {}, producer="test-producer")
    await InboxProcessor(InboxEventRow).wrap(Recorder())(delivered)
    assert (await stored(service, delivered)).topic == topic
