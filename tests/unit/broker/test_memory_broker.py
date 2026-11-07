"""The in-process broker: what it delivers, in what order, and what it drops.

This transport is what the local stack and most tests run on, so its delivery
guarantees are the ones a handler is written against. Each test drains the
queue rather than sleeping for a fixed time, so none of them is timing-shaped.
"""

import asyncio
from typing import Any

import pytest

from base.services.broker.memory import InMemoryBroker
from contracts.events import Event

TOPIC = "test.topic"


def make_broker(*, max_attempts: int = 3) -> InMemoryBroker:
    """A broker with a backoff short enough that a retry costs nothing."""
    return InMemoryBroker(max_attempts=max_attempts, retry_backoff=0.001)


def event(topic: str = TOPIC, **payload: Any) -> Event:
    return Event.of(topic, payload, producer="test-producer")


async def test_a_subscriber_receives_what_was_published() -> None:
    broker = make_broker()
    received: list[Event] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming)

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    published = event(n=1)
    await broker.apublish(published)
    await broker.adrain()
    await broker.astop()
    assert [e.meta.event_id for e in received] == [published.meta.event_id]


async def test_a_handler_subscribed_to_another_topic_hears_nothing() -> None:
    broker = make_broker()
    received: list[Event] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming)

    await broker.asubscribe("other.topic", "group-a", handler)
    await broker.astart()
    await broker.apublish(event(n=1))
    await broker.adrain()
    await broker.astop()
    assert received == []


async def test_every_group_subscribed_to_a_topic_receives_the_event() -> None:
    broker = make_broker()
    first: list[int] = []
    second: list[int] = []

    async def handler_a(incoming: Event) -> None:
        first.append(incoming.payload["n"])

    async def handler_b(incoming: Event) -> None:
        second.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-a", handler_a)
    await broker.asubscribe(TOPIC, "group-b", handler_b)
    await broker.astart()
    await broker.apublish(event(n=1))
    await broker.adrain()
    await broker.astop()
    assert first == [1]
    assert second == [1]


async def test_events_arrive_in_the_order_they_were_published() -> None:
    broker = make_broker()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    for n in range(20):
        await broker.apublish(event(n=n))
    await broker.adrain()
    await broker.astop()
    assert received == list(range(20))


async def test_starting_twice_runs_one_consumer() -> None:
    broker = make_broker()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    task = broker._task
    await broker.astart()
    assert broker._task is task
    await broker.apublish(event(n=1))
    await broker.adrain()
    await broker.astop()
    assert received == [1]


async def test_stopping_twice_is_accepted() -> None:
    broker = make_broker()
    await broker.astart()
    await broker.astop()
    await broker.astop()
    assert broker._task is None


async def test_stopping_without_starting_is_accepted() -> None:
    broker = make_broker()
    await broker.astop()
    assert broker._task is None


async def test_a_stopped_broker_starts_again() -> None:
    broker = make_broker()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    await broker.astop()
    await broker.astart()
    await broker.apublish(event(n=2))
    await broker.adrain()
    await broker.astop()
    assert received == [2]


async def test_an_event_published_while_stopped_is_delivered_on_start() -> None:
    broker = make_broker()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.apublish(event(n=1))
    await asyncio.sleep(0)
    assert received == []
    await broker.astart()
    await broker.adrain()
    await broker.astop()
    assert received == [1]


async def test_stopping_delivers_what_is_still_queued() -> None:
    broker = make_broker()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    for n in range(5):
        await broker.apublish(event(n=n))
    await broker.astop()
    assert received == list(range(5))


async def test_a_handler_subscribed_after_start_receives_later_events() -> None:
    broker = make_broker()
    late: list[int] = []

    async def handler(incoming: Event) -> None:
        late.append(incoming.payload["n"])

    await broker.astart()
    await broker.asubscribe(TOPIC, "group-late", handler)
    await broker.apublish(event(n=1))
    await broker.adrain()
    await broker.astop()
    assert late == [1]


async def test_the_same_event_delivered_twice_runs_a_group_once() -> None:
    broker = make_broker()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    duplicate = event(n=1)
    await broker.apublish(duplicate)
    await broker.apublish(duplicate)
    await broker.adrain()
    await broker.astop()
    assert received == [1]


async def test_two_events_with_the_same_payload_are_both_delivered() -> None:
    """Deduplication keys on the event id, not on what the event says."""
    broker = make_broker()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    await broker.apublish(event(n=1))
    await broker.apublish(event(n=1))
    await broker.adrain()
    await broker.astop()
    assert received == [1, 1]


async def test_a_failing_handler_is_retried_up_to_the_attempt_limit() -> None:
    broker = make_broker(max_attempts=3)
    attempts: list[int] = []

    async def handler(incoming: Event) -> None:
        attempts.append(incoming.payload["n"])
        raise RuntimeError("handler failed")

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    await broker.apublish(event(n=1))
    await broker.adrain()
    await broker.astop()
    assert len(attempts) == 3


async def test_a_handler_that_recovers_is_not_dead_lettered() -> None:
    broker = make_broker(max_attempts=3)
    attempts: list[int] = []

    async def handler(incoming: Event) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first attempt fails")

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    await broker.apublish(event(n=1))
    await broker.adrain()
    await broker.astop()
    assert len(attempts) == 2
    assert broker.dlq_size(TOPIC) == 0


async def test_an_exhausted_event_lands_in_the_dead_letter_queue() -> None:
    broker = make_broker(max_attempts=2)

    async def handler(incoming: Event) -> None:
        raise RuntimeError("handler failed")

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    await broker.apublish(event(n=1))
    await broker.adrain()
    await broker.astop()
    assert broker.dlq_size(TOPIC) == 1


async def test_one_group_failing_does_not_stop_another_from_receiving() -> None:
    broker = make_broker(max_attempts=1)
    healthy: list[int] = []

    async def failing(incoming: Event) -> None:
        raise RuntimeError("handler failed")

    async def working(incoming: Event) -> None:
        healthy.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-bad", failing)
    await broker.asubscribe(TOPIC, "group-good", working)
    await broker.astart()
    await broker.apublish(event(n=1))
    await broker.adrain()
    await broker.astop()
    assert healthy == [1]
    assert broker.dlq_size(TOPIC) == 1


async def test_replaying_the_dead_letter_queue_redelivers_it() -> None:
    broker = make_broker(max_attempts=1)
    attempts: list[int] = []
    fail = True

    async def handler(incoming: Event) -> None:
        attempts.append(1)
        if fail:
            raise RuntimeError("handler failed")

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    await broker.apublish(event(n=1))
    await broker.adrain()
    assert broker.dlq_size(TOPIC) == 1
    fail = False
    assert await broker.areplay_dlq(TOPIC) == 1
    await broker.adrain()
    await broker.astop()
    assert len(attempts) == 2
    assert broker.dlq_size(TOPIC) == 0


async def test_replaying_an_empty_dead_letter_queue_replays_nothing() -> None:
    broker = make_broker()
    assert await broker.areplay_dlq("never.used") == 0


async def test_replaying_honours_the_limit() -> None:
    broker = make_broker(max_attempts=1)

    async def handler(incoming: Event) -> None:
        raise RuntimeError("handler failed")

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    for n in range(3):
        await broker.apublish(event(n=n))
    await broker.adrain()
    await broker.astop()
    assert broker.dlq_size(TOPIC) == 3
    assert await broker.areplay_dlq(TOPIC, limit=2) == 2
    assert broker.dlq_size(TOPIC) == 1


async def test_the_seen_set_is_bounded() -> None:
    broker = make_broker()
    for n in range(InMemoryBroker.SEEN_LIMIT + 10):
        broker._mark_seen(f"group:{n}")
    assert len(broker._seen) == InMemoryBroker.SEEN_LIMIT


async def test_the_oldest_seen_key_is_the_one_dropped() -> None:
    broker = make_broker()
    for n in range(InMemoryBroker.SEEN_LIMIT + 1):
        broker._mark_seen(f"group:{n}")
    assert "group:0" not in broker._seen
    assert f"group:{InMemoryBroker.SEEN_LIMIT}" in broker._seen


async def test_an_in_process_broker_is_always_healthy() -> None:
    assert await make_broker().ahealth_check() is True


async def test_an_event_with_an_empty_payload_is_delivered() -> None:
    broker = make_broker()
    received: list[dict] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload)

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    await broker.apublish(Event.of(TOPIC, {}, producer="test-producer"))
    await broker.adrain()
    await broker.astop()
    assert received == [{}]


async def test_a_very_large_payload_is_delivered_intact() -> None:
    broker = make_broker()
    received: list[dict] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload)

    payload = {"blob": "x" * 1_000_000, "items": list(range(10_000))}
    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    await broker.apublish(Event.of(TOPIC, payload, producer="test-producer"))
    await broker.adrain()
    await broker.astop()
    assert received == [payload]


async def test_a_slow_handler_does_not_lose_what_is_behind_it() -> None:
    broker = make_broker()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        await asyncio.sleep(0.05)
        received.append(incoming.payload["n"])

    await broker.asubscribe(TOPIC, "group-a", handler)
    await broker.astart()
    for n in range(4):
        await broker.apublish(event(n=n))
    await broker.adrain()
    await broker.astop()
    assert received == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "topic",
    [
        "topic with spaces",
        "topic:with:colons",
        "topic/with/slashes",
        "tópico-ünicode-🎉",
        "topic.{braces}",
        "*",
        "a" * 500,
    ],
)
async def test_an_unusual_topic_still_routes(topic: str) -> None:
    broker = make_broker()
    received: list[str] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.topic)

    await broker.asubscribe(topic, "group-a", handler)
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    await broker.adrain()
    await broker.astop()
    assert received == [topic]


async def test_an_empty_topic_still_routes() -> None:
    broker = make_broker()
    received: list[str] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.topic)

    await broker.asubscribe("", "group-a", handler)
    await broker.astart()
    await broker.apublish(event("", n=1))
    await broker.adrain()
    await broker.astop()
    assert received == [""]
