"""The worker end to end: published, consumed, claimed, handled, acked.

The transport is the in-process one and the inbox is the real one on SQLite,
so what is exercised here is the loop a deployment actually runs — including
the two paths that only appear under failure: a bounded retry, and a delivery
that is dead-lettered rather than redelivered forever.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from base.models.definitions.messaging import InboxState
from base.services.broker.memory import InMemoryBroker
from contracts.events import Event
from contracts.v1 import chat, identity
from ingestion_worker import handlers, worker
from tests.unit.ingestion_worker.conftest import (
    PRODUCER,
    conversation_created,
    find,
    row_count,
    stored,
    user_registered,
)


@pytest.fixture
async def consuming(worker_settings: Any) -> AsyncIterator[InMemoryBroker]:
    """A started worker on a transport whose backoff costs nothing."""
    del worker_settings
    broker = InMemoryBroker(max_attempts=3, retry_backoff=0.001)
    await worker.arun(broker)
    yield broker
    await broker.astop()


async def deliver(broker: InMemoryBroker, *events: Event) -> None:
    for event in events:
        await broker.apublish(event)
    await broker.adrain()


def always_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing(incoming: Event) -> None:
        raise RuntimeError("handler failed")

    monkeypatch.setattr(handlers, "ahandle_user_registered", failing)


class Outage:
    """A handler that fails until whatever it depends on comes back."""

    def __init__(self) -> None:
        self.over = False
        self.runs = 0

    async def __call__(self, incoming: Event) -> None:
        self.runs += 1
        if not self.over:
            raise RuntimeError("handler failed")


# The happy path


async def test_a_published_registration_is_handled(
    consuming: InMemoryBroker,
) -> None:
    delivered = user_registered()
    await deliver(consuming, delivered)
    assert handlers.processed_count() == 1
    assert (await stored(delivered)).state == InboxState.PROCESSED


async def test_a_published_conversation_is_handled(
    consuming: InMemoryBroker,
) -> None:
    delivered = conversation_created()
    await deliver(consuming, delivered)
    assert handlers.processed_count() == 1
    assert (await stored(delivered)).state == InboxState.PROCESSED


async def test_both_topics_are_consumed_by_one_worker(
    consuming: InMemoryBroker,
) -> None:
    await deliver(consuming, user_registered(), conversation_created())
    assert handlers.processed_count() == 2
    assert await row_count() == 2


async def test_a_topic_this_worker_does_not_own_is_ignored(
    consuming: InMemoryBroker,
) -> None:
    """Deactivation is published by identity and consumed by nobody here."""
    unsubscribed = Event.of(
        identity.USER_DEACTIVATED,
        {"user_id": "11111111-1111-1111-1111-111111111111"},
        producer=PRODUCER,
    )
    await deliver(consuming, unsubscribed)
    assert handlers.processed_count() == 0
    assert await row_count() == 0


async def test_an_entirely_unknown_topic_is_ignored(
    consuming: InMemoryBroker,
) -> None:
    await deliver(consuming, Event.of("nobody.owns.this", {}, producer=PRODUCER))
    assert await row_count() == 0


async def test_a_backlog_published_before_startup_is_consumed_after_it(
    worker_settings: Any,
) -> None:
    del worker_settings
    broker = InMemoryBroker(max_attempts=3, retry_backoff=0.001)
    waiting = user_registered()
    await broker.apublish(waiting)
    await asyncio.sleep(0)
    assert await find(waiting) is None

    await worker.arun(broker)
    await broker.adrain()
    await broker.astop()
    assert (await stored(waiting)).state == InboxState.PROCESSED


async def test_what_is_still_queued_at_shutdown_is_still_handled(
    worker_settings: Any,
) -> None:
    del worker_settings
    broker = InMemoryBroker(max_attempts=3, retry_backoff=0.001)
    await worker.arun(broker)
    events = [user_registered(user_id=str(n)) for n in range(5)]
    for event in events:
        await broker.apublish(event)
    await broker.astop()
    assert handlers.processed_count() == 5


async def test_many_events_are_each_handled_once(
    consuming: InMemoryBroker,
) -> None:
    events = [user_registered(user_id=str(n)) for n in range(20)]
    await deliver(consuming, *events)
    assert handlers.processed_count() == 20
    assert await row_count() == 20


# Redelivery


async def test_an_event_published_twice_is_handled_once(
    consuming: InMemoryBroker,
) -> None:
    delivered = user_registered()
    await deliver(consuming, delivered, delivered)
    assert handlers.processed_count() == 1
    assert await row_count() == 1


async def test_a_redelivery_the_transport_did_not_deduplicate_is_handled_once(
    consuming: InMemoryBroker,
) -> None:
    """A distributed transport has no memory of what it delivered; the inbox does."""
    delivered = user_registered()
    await deliver(consuming, delivered)
    consuming._seen.clear()
    await deliver(consuming, delivered)
    assert handlers.processed_count() == 1


async def test_two_registrations_saying_the_same_thing_are_both_handled(
    consuming: InMemoryBroker,
) -> None:
    await deliver(consuming, user_registered(), user_registered())
    assert handlers.processed_count() == 2


# Failure, retry and the dead letter queue


async def test_a_failing_handler_is_retried_to_the_limit(
    worker_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    del worker_settings
    always_fails(monkeypatch)
    broker = InMemoryBroker(max_attempts=3, retry_backoff=0.001)
    await worker.arun(broker)
    delivered = user_registered()
    await deliver(broker, delivered)
    await broker.astop()
    assert (await stored(delivered)).attempts == 3


async def test_an_exhausted_delivery_lands_in_the_dead_letter_queue(
    worker_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It has to stop somewhere, or the stream never advances past it."""
    del worker_settings
    always_fails(monkeypatch)
    broker = InMemoryBroker(max_attempts=2, retry_backoff=0.001)
    await worker.arun(broker)
    delivered = user_registered()
    await deliver(broker, delivered)
    await broker.astop()
    assert broker.dlq_size(identity.USER_REGISTERED) == 1


async def test_an_exhausted_delivery_leaves_the_failure_in_the_inbox(
    worker_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    del worker_settings
    always_fails(monkeypatch)
    broker = InMemoryBroker(max_attempts=2, retry_backoff=0.001)
    await worker.arun(broker)
    delivered = user_registered()
    await deliver(broker, delivered)
    await broker.astop()
    row = await stored(delivered)
    assert row.state == InboxState.FAILED
    assert row.attempts == 2
    assert row.last_error == "RuntimeError: handler failed"
    assert row.processed_at is None


async def test_a_handler_that_recovers_is_never_dead_lettered(
    worker_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    del worker_settings
    attempts: list[int] = []

    async def flaky(incoming: Event) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first attempt fails")

    monkeypatch.setattr(handlers, "ahandle_user_registered", flaky)
    broker = InMemoryBroker(max_attempts=3, retry_backoff=0.001)
    await worker.arun(broker)
    delivered = user_registered()
    await deliver(broker, delivered)
    await broker.astop()
    row = await stored(delivered)
    assert len(attempts) == 2
    assert row.state == InboxState.PROCESSED
    assert row.attempts == 2
    assert row.last_error is None
    assert broker.dlq_size(identity.USER_REGISTERED) == 0


async def test_replaying_a_dead_letter_runs_the_handler_that_now_works(
    worker_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the queue: a delivery parked there is work, not waste."""
    del worker_settings
    outage = Outage()
    monkeypatch.setattr(handlers, "ahandle_user_registered", outage)
    broker = InMemoryBroker(max_attempts=2, retry_backoff=0.001)
    await worker.arun(broker)
    delivered = user_registered()
    await deliver(broker, delivered)
    assert broker.dlq_size(identity.USER_REGISTERED) == 1

    outage.over = True
    broker._seen.clear()
    assert await broker.areplay_dlq(identity.USER_REGISTERED) == 1
    await broker.adrain()
    await broker.astop()
    assert (await stored(delivered)).state == InboxState.PROCESSED
    assert broker.dlq_size(identity.USER_REGISTERED) == 0


async def test_one_failing_topic_does_not_stop_the_other(
    worker_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    del worker_settings
    always_fails(monkeypatch)
    broker = InMemoryBroker(max_attempts=2, retry_backoff=0.001)
    await worker.arun(broker)
    healthy = conversation_created()
    await deliver(broker, user_registered(), healthy)
    await broker.astop()
    assert (await stored(healthy)).state == InboxState.PROCESSED
    assert broker.dlq_size(chat.CONVERSATION_CREATED) == 0


# Payloads the producer got wrong


async def test_a_payload_that_does_not_match_its_topic_is_not_a_poison_pill(
    consuming: InMemoryBroker,
) -> None:
    mismatched = Event.of(
        identity.USER_REGISTERED,
        {"conversation_id": "22222222-2222-2222-2222-222222222222"},
        producer=PRODUCER,
    )
    await deliver(consuming, mismatched)
    assert (await stored(mismatched)).state == InboxState.PROCESSED
    assert consuming.dlq_size(identity.USER_REGISTERED) == 0


async def test_an_empty_payload_is_consumed(consuming: InMemoryBroker) -> None:
    empty = Event.of(identity.USER_REGISTERED, {}, producer=PRODUCER)
    await deliver(consuming, empty)
    assert (await stored(empty)).state == InboxState.PROCESSED


async def test_a_very_large_payload_is_consumed(
    consuming: InMemoryBroker,
) -> None:
    large = Event.of(
        chat.CONVERSATION_CREATED, {"blob": "x" * 200_000}, producer=PRODUCER
    )
    await deliver(consuming, large)
    assert (await stored(large)).state == InboxState.PROCESSED
