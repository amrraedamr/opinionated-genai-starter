"""The claim this worker commits before it runs a handler.

Delivery is at-least-once, so the same event arrives more than once and this
table is the only thing standing between that and the work being done twice.
Everything here is about the claim: who gets it, who is turned away, what a
process that died holding one leaves behind, and what the row says afterwards.

The generic processor is pinned against PostgreSQL in
`tests/integration/messaging`; what is pinned here is the worker's own model
on the backend it ships for local use.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from base.models.definitions.messaging import InboxState
from base.models.view_models.exceptions.sql import SQLConflictError
from base.services.messaging import InboxProcessor
from base.services.sql.service import SQLService
from contracts.events import Event
from ingestion_worker.models.inbox import InboxEventDTO
from tests.unit.ingestion_worker.conftest import (
    Recorder,
    conversation_created,
    row_count,
    set_claim,
    stored,
    user_registered,
)


def processor(claim_timeout_seconds: int = 300) -> InboxProcessor:
    return InboxProcessor(InboxEventDTO, claim_timeout_seconds=claim_timeout_seconds)


# A first delivery


async def test_a_first_delivery_runs_the_handler(worker_settings: Any) -> None:
    del worker_settings
    handler = Recorder()
    delivered = user_registered()
    await processor().wrap(handler)(delivered)
    assert [e.meta.event_id for e in handler.runs] == [delivered.meta.event_id]


async def test_a_handled_event_is_recorded_as_processed(
    worker_settings: Any,
) -> None:
    del worker_settings
    delivered = user_registered()
    await processor().wrap(Recorder())(delivered)
    row = await stored(delivered)
    assert row.state == InboxState.PROCESSED
    assert row.topic == delivered.topic
    assert row.attempts == 1
    assert row.processed_at is not None
    assert row.last_error is None


async def test_the_row_records_the_topic_the_event_carried(
    worker_settings: Any,
) -> None:
    del worker_settings
    delivered = conversation_created()
    await processor().wrap(Recorder())(delivered)
    assert (await stored(delivered)).topic == "chat.conversation.created"


async def test_two_different_events_both_run(worker_settings: Any) -> None:
    del worker_settings
    handler = Recorder()
    wrapped = processor().wrap(handler)
    await wrapped(user_registered())
    await wrapped(conversation_created())
    assert handler.count == 2
    assert await row_count() == 2


# Redelivery


async def test_an_event_delivered_twice_is_handled_once(
    worker_settings: Any,
) -> None:
    del worker_settings
    handler = Recorder()
    wrapped = processor().wrap(handler)
    delivered = user_registered()
    await wrapped(delivered)
    await wrapped(delivered)
    assert handler.count == 1
    assert await row_count() == 1


async def test_an_event_delivered_many_times_is_still_handled_once(
    worker_settings: Any,
) -> None:
    del worker_settings
    handler = Recorder()
    wrapped = processor().wrap(handler)
    delivered = user_registered()
    for _ in range(5):
        await wrapped(delivered)
    assert handler.count == 1


async def test_a_redelivery_after_a_restart_is_still_handled_once(
    worker_settings: Any,
) -> None:
    """A new process must not replay what the previous one already finished."""
    del worker_settings
    handler = Recorder()
    delivered = user_registered()
    await processor().wrap(handler)(delivered)
    await processor().wrap(handler)(delivered)
    assert handler.count == 1


async def test_two_events_saying_the_same_thing_are_both_handled(
    worker_settings: Any,
) -> None:
    """Idempotency keys on the event id, not on what the event says."""
    del worker_settings
    handler = Recorder()
    wrapped = processor().wrap(handler)
    await wrapped(user_registered())
    await wrapped(user_registered())
    assert handler.count == 2


async def test_a_duplicate_arriving_while_a_claim_is_in_flight_is_turned_away(
    worker_settings: Any,
) -> None:
    """Two consumers can hold one event at once; only the first may run it."""
    del worker_settings
    handler = Recorder()
    wrapped = processor().wrap(handler)
    delivered = user_registered()
    await wrapped(delivered)
    await set_claim(delivered, state=InboxState.PROCESSING, age_seconds=1)
    await wrapped(delivered)
    assert handler.count == 1


# A handler that raised


async def test_a_failing_handler_is_recorded_as_failed_and_raises(
    worker_settings: Any,
) -> None:
    """The broker must see the failure, or it would ack and never redeliver."""
    del worker_settings
    delivered = user_registered()
    with pytest.raises(ValueError, match="handler failed"):
        await processor().wrap(Recorder(failing=True))(delivered)
    row = await stored(delivered)
    assert row.state == InboxState.FAILED
    assert row.processed_at is None
    assert row.last_error == "ValueError: handler failed"


async def test_a_failed_event_is_claimed_again_on_redelivery(
    worker_settings: Any,
) -> None:
    del worker_settings
    succeeding = Recorder()
    wrapped_failing = processor().wrap(Recorder(failing=True))
    delivered = user_registered()
    with pytest.raises(ValueError):
        await wrapped_failing(delivered)
    await processor().wrap(succeeding)(delivered)
    row = await stored(delivered)
    assert succeeding.count == 1
    assert row.state == InboxState.PROCESSED
    assert row.attempts == 2
    assert row.last_error is None


async def test_every_failed_attempt_is_counted(worker_settings: Any) -> None:
    del worker_settings
    wrapped = processor().wrap(Recorder(failing=True))
    delivered = user_registered()
    for _ in range(3):
        with pytest.raises(ValueError):
            await wrapped(delivered)
    assert (await stored(delivered)).attempts == 3


async def test_a_handler_that_raises_something_other_than_an_error_still_fails(
    worker_settings: Any,
) -> None:
    del worker_settings

    async def handler(incoming: Event) -> None:
        raise KeyError("user_id")

    delivered = user_registered()
    with pytest.raises(KeyError):
        await processor().wrap(handler)(delivered)
    assert (await stored(delivered)).last_error == "KeyError: 'user_id'"


# A process that died holding a claim


async def test_a_crash_between_the_claim_and_the_handler_leaves_the_claim_held(
    worker_settings: Any,
) -> None:
    """The row stays `processing`, which is what makes the work retryable."""
    del worker_settings

    async def dies(incoming: Event) -> None:
        raise asyncio.CancelledError

    delivered = user_registered()
    with pytest.raises(asyncio.CancelledError):
        await processor().wrap(dies)(delivered)
    row = await stored(delivered)
    assert row.state == InboxState.PROCESSING
    assert row.processed_at is None


async def test_work_a_dead_process_never_finished_is_picked_up_again(
    worker_settings: Any,
) -> None:
    del worker_settings

    async def dies(incoming: Event) -> None:
        raise asyncio.CancelledError

    handler = Recorder()
    claimed = processor(claim_timeout_seconds=30)
    delivered = user_registered()
    with pytest.raises(asyncio.CancelledError):
        await claimed.wrap(dies)(delivered)
    await set_claim(delivered, state=InboxState.PROCESSING, age_seconds=600)
    await claimed.wrap(handler)(delivered)
    assert handler.count == 1
    assert (await stored(delivered)).state == InboxState.PROCESSED


async def test_a_claim_older_than_the_timeout_is_taken_over(
    worker_settings: Any,
) -> None:
    del worker_settings
    handler = Recorder()
    claimed = processor(claim_timeout_seconds=30)
    delivered = user_registered()
    await claimed.wrap(handler)(delivered)
    await set_claim(delivered, state=InboxState.PROCESSING, age_seconds=600)
    await claimed.wrap(handler)(delivered)
    assert handler.count == 2


async def test_a_claim_younger_than_the_timeout_is_left_alone(
    worker_settings: Any,
) -> None:
    del worker_settings
    handler = Recorder()
    claimed = processor(claim_timeout_seconds=3_600)
    delivered = user_registered()
    await claimed.wrap(handler)(delivered)
    await set_claim(delivered, state=InboxState.PROCESSING, age_seconds=600)
    await claimed.wrap(handler)(delivered)
    assert handler.count == 1


async def test_taking_over_a_stale_claim_counts_an_attempt(
    worker_settings: Any,
) -> None:
    del worker_settings
    claimed = processor(claim_timeout_seconds=30)
    delivered = user_registered()
    await claimed.wrap(Recorder())(delivered)
    for _ in range(3):
        await set_claim(delivered, state=InboxState.PROCESSING, age_seconds=600)
        await claimed.wrap(Recorder())(delivered)
    assert (await stored(delivered)).attempts == 4


async def test_an_event_already_processed_is_never_taken_over(
    worker_settings: Any,
) -> None:
    """A terminal state outranks the age of the lock that produced it."""
    del worker_settings
    handler = Recorder()
    claimed = processor(claim_timeout_seconds=30)
    delivered = user_registered()
    await claimed.wrap(handler)(delivered)
    await set_claim(delivered, state=InboxState.PROCESSED, age_seconds=10_000)
    await claimed.wrap(handler)(delivered)
    assert handler.count == 1


# Two consumers at once


async def test_the_same_event_id_arriving_concurrently_runs_the_handler_once(
    worker_settings: Any,
) -> None:
    """Exactly one claim wins; the losers see the row or lose the insert."""
    del worker_settings
    handler = Recorder()
    wrapped = processor().wrap(handler)
    delivered = user_registered()

    outcomes = await asyncio.gather(
        *(wrapped(delivered) for _ in range(4)), return_exceptions=True
    )
    raised = [o for o in outcomes if isinstance(o, BaseException)]

    assert handler.count == 1
    assert await row_count() == 1
    assert all(isinstance(error, SQLConflictError) for error in raised)
    assert len(raised) <= 3


async def test_concurrent_distinct_events_all_run(worker_settings: Any) -> None:
    del worker_settings
    handler = Recorder()
    wrapped = processor().wrap(handler)
    events = [user_registered(user_id=str(n)) for n in range(6)]

    await asyncio.gather(*(wrapped(delivered) for delivered in events))
    assert {e.meta.event_id for e in handler.runs} == {e.meta.event_id for e in events}


# What the handler sees while it holds the claim


async def test_a_handler_runs_while_the_row_still_says_processing(
    worker_settings: Any,
) -> None:
    del worker_settings
    seen: list[str] = []

    async def handler(incoming: Event) -> None:
        seen.append((await stored(incoming)).state)

    delivered = user_registered()
    await processor().wrap(handler)(delivered)
    assert seen == [InboxState.PROCESSING]
    assert (await stored(delivered)).state == InboxState.PROCESSED


async def test_a_claim_that_disappears_mid_handler_is_reported(
    worker_settings: Any,
) -> None:
    """Another process truncating the table must not pass silently."""
    del worker_settings

    async def handler(incoming: Event) -> None:
        async with SQLService().atransaction() as session:
            row = await session.get(InboxEventDTO, incoming.meta.event_id)
            await session.delete(row)

    with pytest.raises(RuntimeError, match="Inbox claim disappeared"):
        await processor().wrap(handler)(user_registered())


# Payloads


async def test_an_event_with_an_empty_payload_is_claimed_and_handled(
    worker_settings: Any,
) -> None:
    del worker_settings
    handler = Recorder()
    delivered = Event.of("identity.user.registered", {}, producer="test-producer")
    await processor().wrap(handler)(delivered)
    assert handler.runs[0].payload == {}
    assert (await stored(delivered)).state == InboxState.PROCESSED


async def test_a_payload_that_does_not_match_its_topic_is_still_claimed(
    worker_settings: Any,
) -> None:
    """The inbox keys on the id and the topic, never on what the payload says."""
    del worker_settings
    handler = Recorder()
    delivered = Event.of(
        "identity.user.registered",
        {"conversation_id": "22222222-2222-2222-2222-222222222222"},
        producer="test-producer",
    )
    await processor().wrap(handler)(delivered)
    row = await stored(delivered)
    assert handler.count == 1
    assert row.topic == "identity.user.registered"
    assert row.state == InboxState.PROCESSED


async def test_a_large_payload_is_handled(worker_settings: Any) -> None:
    del worker_settings
    handler = Recorder()
    payload = {"blob": "x" * 200_000}
    delivered = Event.of("identity.user.registered", payload, producer="test-producer")
    await processor().wrap(handler)(delivered)
    assert handler.runs[0].payload == payload
