"""The outbox against the real database.

The point of the pattern is that a domain write and its event share one
transaction, so the first test here is the one that matters: a rolled back
write leaves no event behind to publish. The rest pins what the dispatcher
does with the rows that did commit, including the ones it cannot publish.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from base.services.messaging import OutboxDispatcher, add_event
from base.services.sql.service import SQLService
from contracts.events import Event
from tests.integration.messaging.conftest import (
    CollectingBroker,
    FailingBroker,
    OutboxEventRow,
    event,
)

pytestmark = pytest.mark.integration


async def row_count(service: SQLService) -> int:
    async with service.atransaction() as session:
        return int(
            await session.scalar(select(func.count()).select_from(OutboxEventRow)) or 0
        )


async def stored(service: SQLService, published: Event) -> OutboxEventRow:
    async with service.atransaction() as session:
        row = await session.get(OutboxEventRow, published.meta.event_id)
        assert row is not None
        return row


async def make_available(service: SQLService, published: Event) -> None:
    """Bring a backed-off row forward, so a test need not wait out the backoff."""
    async with service.atransaction() as session:
        row = await session.get(OutboxEventRow, published.meta.event_id)
        assert row is not None
        row.available_at = datetime.now(UTC) - timedelta(seconds=1)


async def test_a_rolled_back_write_leaves_no_event(messaging_tables: Any) -> None:
    """The whole point of the pattern, stated as plainly as it can be."""
    service = messaging_tables
    with pytest.raises(RuntimeError, match="the domain write failed"):
        async with service.atransaction() as session:
            add_event(session, OutboxEventRow, event())
            raise RuntimeError("the domain write failed")
    assert await row_count(service) == 0


async def test_a_committed_write_keeps_its_event(messaging_tables: Any) -> None:
    service = messaging_tables
    published = event("user.registered", id=7)
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)
    row = await stored(service, published)
    assert row.topic == "user.registered"
    assert Event.model_validate(row.event).payload == {"id": 7}


async def test_a_fresh_row_is_unpublished_and_untried(messaging_tables: Any) -> None:
    service = messaging_tables
    published = event()
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)
    row = await stored(service, published)
    assert row.published_at is None
    assert row.attempts == 0
    assert row.last_error is None


async def test_dispatching_publishes_what_committed(messaging_tables: Any) -> None:
    service = messaging_tables
    published = event("user.registered", id=7)
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    broker = CollectingBroker()
    assert await OutboxDispatcher(broker, OutboxEventRow).adispatch_once() == 1  # type: ignore[arg-type]
    assert [e.meta.event_id for e in broker.published] == [published.meta.event_id]
    assert broker.published[0].payload == {"id": 7}


async def test_a_published_row_is_marked_and_not_sent_again(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    published = event()
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    broker = CollectingBroker()
    dispatcher = OutboxDispatcher(broker, OutboxEventRow)  # type: ignore[arg-type]
    await dispatcher.adispatch_once()
    row = await stored(service, published)
    assert row.published_at is not None
    assert await dispatcher.adispatch_once() == 0
    assert len(broker.published) == 1


async def test_dispatching_an_empty_outbox_claims_nothing(
    messaging_tables: Any,
) -> None:
    del messaging_tables
    broker = CollectingBroker()
    assert await OutboxDispatcher(broker, OutboxEventRow).adispatch_once() == 0  # type: ignore[arg-type]
    assert broker.published == []


async def test_events_are_published_in_the_order_they_occurred(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    occurred = datetime.now(UTC)
    events = [event(n=n) for n in range(5)]
    async with service.atransaction() as session:
        for offset, published in enumerate(reversed(events)):
            published.meta.occurred_at = occurred + timedelta(seconds=offset)
            add_event(session, OutboxEventRow, published)

    broker = CollectingBroker()
    await OutboxDispatcher(broker, OutboxEventRow).adispatch_once()  # type: ignore[arg-type]
    assert [e.payload["n"] for e in broker.published] == [4, 3, 2, 1, 0]


async def test_a_batch_is_capped_at_the_configured_size(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    async with service.atransaction() as session:
        for n in range(5):
            add_event(session, OutboxEventRow, event(n=n))

    broker = CollectingBroker()
    dispatcher = OutboxDispatcher(broker, OutboxEventRow, batch_size=2)  # type: ignore[arg-type]
    assert await dispatcher.adispatch_once() == 2
    assert await dispatcher.adispatch_once() == 2
    assert await dispatcher.adispatch_once() == 1
    assert await dispatcher.adispatch_once() == 0
    assert len(broker.published) == 5


async def test_a_failed_publication_leaves_the_row_for_another_go(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    published = event()
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    assert await OutboxDispatcher(FailingBroker(), OutboxEventRow).adispatch_once() == 1  # type: ignore[arg-type]
    row = await stored(service, published)
    assert row.published_at is None
    assert row.attempts == 1
    assert row.last_error == "RuntimeError: transport unavailable"


async def test_a_failed_row_is_held_back_until_its_backoff_elapses(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    published = event()
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    broker = FailingBroker()
    dispatcher = OutboxDispatcher(broker, OutboxEventRow)  # type: ignore[arg-type]
    await dispatcher.adispatch_once()
    assert (await stored(service, published)).available_at > datetime.now(UTC)
    assert await dispatcher.adispatch_once() == 0
    assert broker.attempts == 1


async def test_each_failure_lengthens_the_backoff(messaging_tables: Any) -> None:
    service = messaging_tables
    published = event()
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    dispatcher = OutboxDispatcher(FailingBroker(), OutboxEventRow)  # type: ignore[arg-type]
    delays: list[float] = []
    for _ in range(4):
        await make_available(service, published)
        before = datetime.now(UTC)
        await dispatcher.adispatch_once()
        row = await stored(service, published)
        delays.append((row.available_at - before).total_seconds())
    assert row.attempts == 4
    assert [round(delay) for delay in delays] == [1, 2, 4, 8]


async def test_the_backoff_never_exceeds_the_cap(messaging_tables: Any) -> None:
    """Otherwise a long outage would park a row for years."""
    service = messaging_tables
    published = event()
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    async with service.atransaction() as session:
        row = await session.get(OutboxEventRow, published.meta.event_id)
        assert row is not None
        row.attempts = 40

    await make_available(service, published)
    before = datetime.now(UTC)
    await OutboxDispatcher(FailingBroker(), OutboxEventRow).adispatch_once()  # type: ignore[arg-type]
    row = await stored(service, published)
    assert round((row.available_at - before).total_seconds()) == 60


async def test_a_row_is_never_abandoned_however_often_it_fails(
    messaging_tables: Any,
) -> None:
    """Pins a gap: the outbox has no attempt limit and no parking.

    `adispatch_once` grows the backoff and records the error, but nothing ever
    takes a row out of the queue, so a permanently unpublishable event is
    retried every sixty seconds for as long as the service runs. There is no
    dead letter table and no terminal state on this side of the broker.
    `base/services/messaging/outbox.py:101`.
    """
    service = messaging_tables
    published = event()
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    dispatcher = OutboxDispatcher(FailingBroker(), OutboxEventRow)  # type: ignore[arg-type]
    for _ in range(12):
        await make_available(service, published)
        assert await dispatcher.adispatch_once() == 1
    row = await stored(service, published)
    assert row.attempts == 12
    assert row.published_at is None


async def test_a_row_recovers_once_the_broker_returns(messaging_tables: Any) -> None:
    service = messaging_tables
    published = event()
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    await OutboxDispatcher(FailingBroker(), OutboxEventRow).adispatch_once()  # type: ignore[arg-type]
    await make_available(service, published)

    broker = CollectingBroker()
    assert await OutboxDispatcher(broker, OutboxEventRow).adispatch_once() == 1  # type: ignore[arg-type]
    row = await stored(service, published)
    assert row.published_at is not None
    assert row.last_error is None
    assert len(broker.published) == 1


async def test_one_failure_does_not_block_the_rest_of_the_batch(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    first, second = event(n=1), event(n=2)
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, first)
        add_event(session, OutboxEventRow, second)

    class FlakyBroker:
        def __init__(self) -> None:
            self.published: list[Event] = []

        async def apublish(self, published: Event) -> None:
            if published.meta.event_id == first.meta.event_id:
                raise RuntimeError("just this one")
            self.published.append(published)

    broker = FlakyBroker()
    assert await OutboxDispatcher(broker, OutboxEventRow).adispatch_once() == 2  # type: ignore[arg-type]
    assert [e.payload["n"] for e in broker.published] == [2]
    assert (await stored(service, first)).published_at is None
    assert (await stored(service, second)).published_at is not None


async def test_the_error_recorded_is_truncated(messaging_tables: Any) -> None:
    service = messaging_tables
    published = event()
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    broker = FailingBroker(RuntimeError("x" * 5_000))
    await OutboxDispatcher(broker, OutboxEventRow).adispatch_once()  # type: ignore[arg-type]
    assert len((await stored(service, published)).last_error or "") == 1_000


async def test_an_event_with_an_empty_payload_round_trips(
    messaging_tables: Any,
) -> None:
    service = messaging_tables
    published = Event.of("test.messaging", {}, producer="test-producer")
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    broker = CollectingBroker()
    await OutboxDispatcher(broker, OutboxEventRow).adispatch_once()  # type: ignore[arg-type]
    assert broker.published[0].payload == {}


async def test_a_large_payload_round_trips(messaging_tables: Any) -> None:
    service = messaging_tables
    payload = {"blob": "x" * 200_000, "items": list(range(2_000))}
    published = Event.of("test.messaging", payload, producer="test-producer")
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    broker = CollectingBroker()
    await OutboxDispatcher(broker, OutboxEventRow).adispatch_once()  # type: ignore[arg-type]
    assert broker.published[0].payload == payload


async def test_a_unicode_payload_round_trips(messaging_tables: Any) -> None:
    service = messaging_tables
    payload = {"text": "héllo wörld — 🎉 日本語"}
    published = Event.of("test.messaging", payload, producer="test-producer")
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    broker = CollectingBroker()
    await OutboxDispatcher(broker, OutboxEventRow).adispatch_once()  # type: ignore[arg-type]
    assert broker.published[0].payload == payload


async def test_the_correlation_id_survives_the_outbox(messaging_tables: Any) -> None:
    from uuid import uuid4

    service = messaging_tables
    correlation = uuid4()
    published = Event.of(
        "test.messaging", {}, producer="test-producer", correlation_id=correlation
    )
    async with service.atransaction() as session:
        add_event(session, OutboxEventRow, published)

    broker = CollectingBroker()
    await OutboxDispatcher(broker, OutboxEventRow).adispatch_once()  # type: ignore[arg-type]
    assert broker.published[0].meta.correlation_id == correlation
