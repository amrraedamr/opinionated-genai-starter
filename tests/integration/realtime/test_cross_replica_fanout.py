"""Two replicas, one Redis, and a message that has to reach both.

This is the test the feature exists for. A manager with an in-process cache
passes every unit test and still fails in production the moment there is a
second replica, because the only socket it can reach is one it happens to
hold. Nothing but two managers sharing a real Redis can tell the difference.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from base.clients.redis_client import RedisClient
from base.models.definitions.realtime import (
    REALTIME_PRESENCE_NAMESPACE,
    ChannelDelivery,
    RealtimeMessageType,
)
from base.models.view_models.realtime import ConnectionTicket, RealtimeEnvelope
from base.services.cache.redis import RedisCacheService
from base.services.common.meta import Singleton
from base.services.rate_limiting.memory import InMemoryRateLimiter
from base.services.realtime.connection import RealtimeConnection
from base.services.realtime.manager import ConnectionManager

pytestmark = pytest.mark.integration

USER = str(uuid4())
SETTLE = 2.0


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)

    def frames(self) -> list[RealtimeEnvelope]:
        return [RealtimeEnvelope.model_validate_json(raw) for raw in self.sent]


def replica() -> ConnectionManager:
    """A manager that is genuinely a different object, as a second process is.

    Dropping the cached singleton is what makes this a second replica rather
    than the same one answering twice — without it the test proves nothing.
    """
    Singleton._instances.pop(ConnectionManager, None)
    return ConnectionManager(RedisCacheService())


def connection(device: str, user: str = USER) -> tuple[RealtimeConnection, FakeSocket]:
    socket = FakeSocket()
    return (
        RealtimeConnection(
            socket,  # type: ignore[arg-type]
            ConnectionTicket(
                ticket=f"t-{device}",
                expires_at=datetime.now(UTC) + timedelta(seconds=30),
                subject="user-1",
                user_id=user,
                device_id=device,
            ),
            limiter=InMemoryRateLimiter(),
        ),
        socket,
    )


@pytest.fixture
async def replicas(infra_settings: Any) -> AsyncIterator[tuple[Any, Any]]:
    """Two managers over one Redis, both listening."""
    del infra_settings
    first, second = replica(), replica()
    assert first is not second, "the test is not proving anything"
    await first.astart()
    await second.astart()
    await asyncio.sleep(0.5)
    try:
        yield first, second
    finally:
        await first.adrain()
        await second.adrain()
        client = RedisClient()
        await client.async_client.delete(f"{REALTIME_PRESENCE_NAMESPACE}:{USER}")
        await client.close_all()


async def test_a_message_reaches_a_device_on_another_replica(
    replicas: tuple[Any, Any],
) -> None:
    """The whole point: the replica publishing holds none of the sockets."""
    sender, holder = replicas
    held, socket = connection("phone")
    await holder.aregister(held)

    await sender.apublish_to_user(USER, RealtimeEnvelope(payload={"n": 1}))
    await asyncio.sleep(SETTLE)

    assert any(f.payload == {"n": 1} for f in socket.frames())


async def test_every_device_on_every_replica_is_reached(
    replicas: tuple[Any, Any],
) -> None:
    """A laptop here and a phone there is the normal case, not an edge one."""
    first, second = replicas
    laptop, laptop_socket = connection("laptop")
    phone, phone_socket = connection("phone")
    await first.aregister(laptop)
    await second.aregister(phone)

    await first.apublish_to_user(USER, RealtimeEnvelope(payload={"n": 2}))
    await asyncio.sleep(SETTLE)

    assert any(f.payload == {"n": 2} for f in laptop_socket.frames())
    assert any(f.payload == {"n": 2} for f in phone_socket.frames())


async def test_the_publishing_replica_delivers_to_itself_too(
    replicas: tuple[Any, Any],
) -> None:
    """Delivery is one path. A local shortcut would be a second to keep in step."""
    first, _ = replicas
    held, socket = connection("laptop")
    await first.aregister(held)

    await first.apublish_to_user(USER, RealtimeEnvelope(payload={"n": 3}))
    await asyncio.sleep(SETTLE)

    assert any(f.payload == {"n": 3} for f in socket.frames())


async def test_a_message_is_not_delivered_twice(
    replicas: tuple[Any, Any],
) -> None:
    """One path means one copy, which is what makes the single path worth it."""
    first, _ = replicas
    held, socket = connection("laptop")
    await first.aregister(held)

    await first.apublish_to_user(USER, RealtimeEnvelope(payload={"n": 4}))
    await asyncio.sleep(SETTLE)

    assert len([f for f in socket.frames() if f.payload == {"n": 4}]) == 1


async def test_another_users_devices_are_left_alone(
    replicas: tuple[Any, Any],
) -> None:
    first, second = replicas
    mine, my_socket = connection("laptop")
    other = str(uuid4())
    theirs, their_socket = connection("laptop", user=other)
    await first.aregister(mine)
    await second.aregister(theirs)

    await first.apublish_to_user(USER, RealtimeEnvelope(payload={"n": 5}))
    await asyncio.sleep(SETTLE)

    try:
        assert any(f.payload == {"n": 5} for f in my_socket.frames())
        assert their_socket.sent == []
    finally:
        await RedisClient().async_client.delete(
            f"{REALTIME_PRESENCE_NAMESPACE}:{other}"
        )


async def test_presence_is_visible_from_the_other_replica(
    replicas: tuple[Any, Any],
) -> None:
    """A user on another replica is online; a local registry would say otherwise."""
    first, second = replicas
    await first.aregister(connection("laptop")[0])

    presence = await second.apresence(USER)
    assert presence.online
    assert presence.last_seen is not None


async def test_draining_one_replica_leaves_the_other_serving(
    replicas: tuple[Any, Any],
) -> None:
    """A rolling restart takes replicas one at a time, not the service."""
    first, second = replicas
    leaving, leaving_socket = connection("laptop")
    staying, staying_socket = connection("phone")
    await first.aregister(leaving)
    await second.aregister(staying)

    await first.adrain("Rolling restart")
    assert leaving_socket.closed is not None

    await second.apublish_to_user(USER, RealtimeEnvelope(payload={"n": 6}))
    await asyncio.sleep(SETTLE)
    assert any(f.payload == {"n": 6} for f in staying_socket.frames())


# Durable delivery across replicas


async def test_a_device_reconnecting_to_another_replica_still_gets_its_gap(
    replicas: tuple[Any, Any],
) -> None:
    """The reason the inbox is in Redis and not in the process.

    A load balancer sends a reconnect wherever it likes, which is rarely the
    replica that held the socket before. An in-process inbox would leave the
    device catching up against a replica that never saw the messages, and it
    would receive nothing at all.
    """
    first, second = replicas
    user = str(uuid4())
    for _ in range(3):
        await first.apublish_to_user(
            user,
            RealtimeEnvelope(type=RealtimeMessageType.MESSAGE, payload={}),
            ChannelDelivery.DURABLE,
        )
    await asyncio.sleep(SETTLE)

    phone, socket = connection("phone", user)
    await second.aregister(phone)
    sent = await second.aresume(phone, last_seq=1)

    assert sent == 2
    assert [f.seq for f in socket.frames()] == [2, 3]
    await first._inbox.aclear(user)


async def test_an_acknowledgement_on_one_replica_is_seen_by_the_other(
    replicas: tuple[Any, Any],
) -> None:
    """Cursors are shared too, or a reconnect elsewhere replays what was read."""
    first, second = replicas
    user = str(uuid4())
    await first.apublish_to_user(
        user,
        RealtimeEnvelope(type=RealtimeMessageType.MESSAGE, payload={}),
        ChannelDelivery.DURABLE,
    )
    laptop, _ = connection("laptop", user)

    await first.aacknowledge(laptop, 1)

    assert await second.acursor_of(laptop) == 1
    await first._inbox.aclear(user)


async def test_two_replicas_publishing_at_once_never_reuse_a_sequence(
    replicas: tuple[Any, Any],
) -> None:
    """Two messages sharing a number would make every cursor ambiguous."""
    first, second = replicas
    user = str(uuid4())

    stored = await asyncio.gather(
        *(
            (first if n % 2 == 0 else second).apublish_to_user(
                user,
                RealtimeEnvelope(type=RealtimeMessageType.MESSAGE, payload={"n": n}),
                ChannelDelivery.DURABLE,
            )
            for n in range(12)
        )
    )

    assert sorted(m.seq for m in stored) == list(range(1, 13))
    await first._inbox.aclear(user)
