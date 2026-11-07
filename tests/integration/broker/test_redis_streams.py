"""The Redis Streams broker against the real server.

This is the transport that actually carries events between containers, so
what is pinned here is the part the in-process broker cannot show: consumer
groups, acknowledgement, redelivery of what was never acked, and the dead
letter stream a message ends up in when a handler never succeeds.
"""

import asyncio
from typing import Any

import pytest
from redis.asyncio import Redis

from contracts.events import Event
from tests.integration.broker.conftest import REDIS_URL, StreamsHarness, await_until

pytestmark = pytest.mark.integration


def event(topic: str, **payload: Any) -> Event:
    return Event.of(topic, payload, producer="test-producer")


async def dlq_entries(topic: str) -> list[Any]:
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        return await client.xrange(f"{topic}.dlq", min="-", max="+")
    finally:
        await client.aclose()


async def pending_count(topic: str, group: str) -> int:
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        return int((await client.xpending(topic, group))["pending"])
    finally:
        await client.aclose()


async def test_a_published_event_reaches_the_group(streams: StreamsHarness) -> None:
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    received: list[Event] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming)

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    published = event(topic, n=1)
    await broker.apublish(published)
    assert await await_until(lambda: len(received) == 1)
    assert received[0].meta.event_id == published.meta.event_id
    assert received[0].payload == {"n": 1}


async def test_an_event_published_before_the_group_existed_is_delivered(
    streams: StreamsHarness,
) -> None:
    """The group is created at id `0`, so it reads the stream from the start."""
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.apublish(event(topic, n=1))
    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    assert await await_until(lambda: received == [1])


async def test_a_delivered_event_is_acknowledged(streams: StreamsHarness) -> None:
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    assert await await_until(lambda: received == [1])
    await asyncio.sleep(0.1)
    assert await pending_count(topic, group) == 0


async def test_events_arrive_in_the_order_they_were_published(
    streams: StreamsHarness,
) -> None:
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    for n in range(10):
        await broker.apublish(event(topic, n=n))
    assert await await_until(lambda: len(received) == 10)
    assert received == list(range(10))


async def test_two_groups_both_receive_the_same_event(
    streams: StreamsHarness,
) -> None:
    broker = streams.new()
    topic = streams.topic()
    first: list[int] = []
    second: list[int] = []

    async def handler_a(incoming: Event) -> None:
        first.append(incoming.payload["n"])

    async def handler_b(incoming: Event) -> None:
        second.append(incoming.payload["n"])

    await broker.asubscribe(topic, streams.group("a"), handler_a)
    await broker.asubscribe(topic, streams.group("b"), handler_b)
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    assert await await_until(lambda: first == [1] and second == [1])


async def test_two_consumers_in_one_group_split_the_messages(
    streams: StreamsHarness,
) -> None:
    """No message reaches both, and none is left behind."""
    streams.configure(claim_idle_ms=600_000)
    first_broker, second_broker = streams.new(), streams.new()
    topic, group = streams.topic(), streams.group()
    first: list[int] = []
    second: list[int] = []

    async def handler_a(incoming: Event) -> None:
        await asyncio.sleep(0.02)
        first.append(incoming.payload["n"])

    async def handler_b(incoming: Event) -> None:
        await asyncio.sleep(0.02)
        second.append(incoming.payload["n"])

    await first_broker.asubscribe(topic, group, handler_a)
    await second_broker.asubscribe(topic, group, handler_b)
    await first_broker.astart()
    await second_broker.astart()
    for n in range(12):
        await first_broker.apublish(event(topic, n=n))
        await asyncio.sleep(0.01)
    assert await await_until(lambda: len(first) + len(second) == 12)
    assert sorted(first + second) == list(range(12))
    assert set(first) & set(second) == set()


async def test_the_two_consumers_take_different_names(
    streams: StreamsHarness,
) -> None:
    assert streams.new()._consumer != streams.new()._consumer


async def test_an_unacked_message_is_redelivered(streams: StreamsHarness) -> None:
    """A handler that raises never acks, so the claim loop hands it back."""
    streams.configure(max_attempts=50, claim_idle_ms=0, retry_backoff=0.01)
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    attempts: list[int] = []

    async def handler(incoming: Event) -> None:
        attempts.append(incoming.payload["n"])
        raise RuntimeError("handler failed")

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    assert await await_until(lambda: len(attempts) >= 3)
    assert set(attempts) == {1}


async def test_a_handler_that_recovers_acknowledges_the_message(
    streams: StreamsHarness,
) -> None:
    streams.configure(max_attempts=50, claim_idle_ms=0, retry_backoff=0.01)
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    attempts: list[int] = []

    async def handler(incoming: Event) -> None:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("not yet")

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    assert await await_until(lambda: len(attempts) >= 3)
    await asyncio.sleep(0.2)
    assert await pending_count(topic, group) == 0
    assert await dlq_entries(topic) == []


async def test_a_message_that_never_succeeds_lands_in_the_dead_letter_stream(
    streams: StreamsHarness,
) -> None:
    streams.configure(max_attempts=1, claim_idle_ms=0, retry_backoff=0.01)
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    attempts: list[int] = []

    async def handler(incoming: Event) -> None:
        attempts.append(1)
        raise ValueError("handler failed")

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    published = event(topic, n=1)
    await broker.apublish(published)
    assert await await_until(lambda: len(attempts) >= 1)
    entries: list[Any] = []
    for _ in range(100):
        entries = await dlq_entries(topic)
        if entries:
            break
        await asyncio.sleep(0.05)
    assert len(entries) == 1
    _message_id, fields = entries[0]
    assert fields["source_topic"] == topic
    assert fields["consumer_group"] == group
    assert fields["error_type"] == "ValueError"
    assert "handler failed" in fields["error"]
    assert int(fields["attempts"]) >= 1
    assert Event.model_validate_json(fields["event"]).meta.event_id == (
        published.meta.event_id
    )


async def test_a_dead_lettered_message_is_acknowledged(
    streams: StreamsHarness,
) -> None:
    """Otherwise the claim loop would redeliver it forever after the DLQ copy."""
    streams.configure(max_attempts=1, claim_idle_ms=0, retry_backoff=0.01)
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    attempts: list[int] = []

    async def handler(incoming: Event) -> None:
        attempts.append(1)
        raise ValueError("handler failed")

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    assert await await_until(lambda: len(attempts) >= 1)
    await asyncio.sleep(0.3)
    assert await pending_count(topic, group) == 0
    assert len(attempts) == 1


async def test_an_exhausted_message_takes_the_configured_number_of_attempts(
    streams: StreamsHarness,
) -> None:
    streams.configure(max_attempts=3, claim_idle_ms=0, retry_backoff=0.01)
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    attempts: list[int] = []

    async def handler(incoming: Event) -> None:
        attempts.append(1)
        raise ValueError("handler failed")

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    entries: list[Any] = []
    for _ in range(200):
        entries = await dlq_entries(topic)
        if entries:
            break
        await asyncio.sleep(0.05)
    assert len(entries) == 1
    assert len(attempts) == 3
    assert int(entries[0][1]["attempts"]) == 3


async def test_replaying_the_dead_letter_stream_republishes_and_empties_it(
    streams: StreamsHarness,
) -> None:
    streams.configure(max_attempts=1, claim_idle_ms=0, retry_backoff=0.01)
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    attempts: list[int] = []
    failing = True

    async def handler(incoming: Event) -> None:
        attempts.append(1)
        if failing:
            raise ValueError("handler failed")

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    for _ in range(200):
        if await dlq_entries(topic):
            break
        await asyncio.sleep(0.05)
    assert len(await dlq_entries(topic)) == 1

    failing = False
    assert await broker.areplay_dlq(topic) == 1
    assert await await_until(lambda: len(attempts) >= 2)
    assert await dlq_entries(topic) == []


async def test_replaying_an_empty_dead_letter_stream_replays_nothing(
    streams: StreamsHarness,
) -> None:
    broker = streams.new()
    assert await broker.areplay_dlq(streams.topic("never-used")) == 0


async def test_replaying_honours_the_limit(streams: StreamsHarness) -> None:
    broker = streams.new()
    topic = streams.topic()
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        for n in range(3):
            await client.xadd(
                f"{topic}.dlq", {"event": event(topic, n=n).model_dump_json()}
            )
    finally:
        await client.aclose()
    assert await broker.areplay_dlq(topic, limit=2) == 2
    assert len(await dlq_entries(topic)) == 1


async def test_a_dead_letter_entry_without_an_event_is_skipped(
    streams: StreamsHarness,
) -> None:
    broker = streams.new()
    topic = streams.topic()
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await client.xadd(f"{topic}.dlq", {"source_topic": topic})
    finally:
        await client.aclose()
    assert await broker.areplay_dlq(topic) == 0
    assert len(await dlq_entries(topic)) == 1


async def test_a_reachable_server_reports_healthy(streams: StreamsHarness) -> None:
    assert await streams.new().ahealth_check() is True


async def test_starting_twice_does_not_double_the_consumers(
    streams: StreamsHarness,
) -> None:
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.astart()
    assert len(broker._tasks) == 1
    await broker.apublish(event(topic, n=1))
    assert await await_until(lambda: received == [1])


async def test_stopping_releases_the_connection(streams: StreamsHarness) -> None:
    broker = streams.new()
    await broker.asubscribe(streams.topic(), streams.group(), _noop)
    await broker.astart()
    await broker.astop()
    assert broker._tasks == []
    assert broker._redis is None


async def test_a_stopped_broker_starts_again(streams: StreamsHarness) -> None:
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload["n"])

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.astop()
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    assert await await_until(lambda: received == [1])


async def test_a_subscription_made_after_start_is_ignored_until_a_restart(
    streams: StreamsHarness,
) -> None:
    """Pins a trap the in-process broker does not share.

    `_astart` returns early once it holds tasks and only ever iterates the
    subscriptions it found then, so a handler registered after startup
    consumes nothing until the broker is stopped and started again.
    `base/services/broker/redis_streams.py:54`.
    """
    broker = streams.new()
    early_topic, late_topic = streams.topic("early"), streams.topic("late")
    group = streams.group()
    late: list[int] = []

    async def handler(incoming: Event) -> None:
        late.append(incoming.payload["n"])

    await broker.asubscribe(early_topic, group, _noop)
    await broker.astart()
    await broker.asubscribe(late_topic, group, handler)
    await broker.apublish(event(late_topic, n=1))
    await asyncio.sleep(0.5)
    assert late == []

    await broker.astop()
    await broker.astart()
    assert await await_until(lambda: late == [1])


async def test_an_event_with_an_empty_payload_survives_the_transport(
    streams: StreamsHarness,
) -> None:
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    received: list[dict] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload)

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(Event.of(topic, {}, producer="test-producer"))
    assert await await_until(lambda: received == [{}])


async def test_a_large_payload_survives_the_transport(
    streams: StreamsHarness,
) -> None:
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    payload = {"blob": "x" * 500_000, "items": list(range(5_000))}
    received: list[dict] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload)

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(Event.of(topic, payload, producer="test-producer"))
    assert await await_until(lambda: received == [payload], timeout=20)


async def test_a_unicode_payload_survives_the_transport(
    streams: StreamsHarness,
) -> None:
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    payload = {"text": "héllo wörld — 🎉 日本語"}
    received: list[dict] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.payload)

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(Event.of(topic, payload, producer="test-producer"))
    assert await await_until(lambda: received == [payload])


async def test_the_correlation_id_reaches_the_consumer(
    streams: StreamsHarness,
) -> None:
    from uuid import uuid4

    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    correlation = uuid4()
    received: list[Event] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming)

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(
        Event.of(topic, {}, producer="test-producer", correlation_id=correlation)
    )
    assert await await_until(lambda: len(received) == 1)
    assert received[0].meta.correlation_id == correlation


@pytest.mark.parametrize(
    "suffix",
    ["with spaces", "with:colons", "with/slashes", "ünicode-🎉", "{braces}", "*"],
)
async def test_an_unusual_topic_still_routes(
    streams: StreamsHarness, suffix: str, unique: str
) -> None:
    broker = streams.new()
    topic = streams.track(f"test.{suffix}.{unique}")
    group = streams.group()
    received: list[str] = []

    async def handler(incoming: Event) -> None:
        received.append(incoming.topic)

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    await broker.apublish(event(topic, n=1))
    assert await await_until(lambda: received == [topic])


async def test_a_slow_handler_does_not_lose_what_is_behind_it(
    streams: StreamsHarness,
) -> None:
    broker = streams.new()
    topic, group = streams.topic(), streams.group()
    received: list[int] = []

    async def handler(incoming: Event) -> None:
        await asyncio.sleep(0.1)
        received.append(incoming.payload["n"])

    await broker.asubscribe(topic, group, handler)
    await broker.astart()
    for n in range(5):
        await broker.apublish(event(topic, n=n))
    assert await await_until(lambda: received == list(range(5)), timeout=20)


async def _noop(event: Event) -> None:
    return None
