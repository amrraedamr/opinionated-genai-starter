"""The registry on one replica, and what it does about the others.

A user's laptop can be held by one process and their phone by another, and the
replica handling the request that produces a message is almost never the one
holding the sockets it should reach. These pin the local half — who is
registered, presence, replacing a reconnect, draining — against an in-process
cache. The half that needs a real Redis is in `tests/integration`.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from base.models.definitions.realtime import (
    REALTIME_PRESENCE_NAMESPACE,
    ChannelDelivery,
    RealtimeCloseCode,
    RealtimeMessageType,
)
from base.models.view_models.realtime import ConnectionTicket, RealtimeEnvelope
from base.services.cache.memory import InMemoryCacheService
from base.services.configuration.mixins import RateLimitSettingsMixin
from base.services.rate_limiting.memory import InMemoryRateLimiter
from base.services.realtime.connection import RealtimeConnection
from base.services.realtime.manager import ConnectionManager
from tests.conftest import MakeSettings

USER = uuid4()


class FakeSocket:
    """Records what was sent and whether it was closed."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None
        self.scope: dict[str, Any] = {}

    async def send_text(self, text: str) -> None:
        if self.closed is not None:
            raise ConnectionResetError("closed")
        self.sent.append(text)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)

    def frames(self) -> list[RealtimeEnvelope]:
        return [RealtimeEnvelope.model_validate_json(raw) for raw in self.sent]


def ticket(device: str, user: Any = USER) -> ConnectionTicket:
    return ConnectionTicket(
        ticket=f"t-{device}",
        expires_at=datetime.now(UTC) + timedelta(seconds=30),
        subject="user-1",
        user_id=user,
        device_id=device,
    )


@pytest.fixture
def manager(make_settings: MakeSettings) -> ConnectionManager:
    make_settings(RateLimitSettingsMixin)
    return ConnectionManager(InMemoryCacheService())


def envelope(kind: RealtimeMessageType) -> RealtimeEnvelope:
    return RealtimeEnvelope(type=kind, payload={})


def connect(device: str, user: Any = USER) -> tuple[RealtimeConnection, FakeSocket]:
    socket = FakeSocket()
    limiter = InMemoryRateLimiter()
    return (
        RealtimeConnection(socket, ticket(device, user), limiter=limiter),  # type: ignore[arg-type]
        socket,
    )


async def test_a_registered_device_is_known_to_this_replica(
    manager: ConnectionManager,
) -> None:
    connection, _ = connect("laptop")
    assert await manager.aregister(connection)
    assert manager.devices_of(str(USER)) == ["laptop"]


async def test_a_user_can_hold_several_devices(
    manager: ConnectionManager,
) -> None:
    """A laptop and a phone on one account are two delivery targets."""
    for device in ("laptop", "phone"):
        await manager.aregister(connect(device)[0])
    assert sorted(manager.devices_of(str(USER))) == ["laptop", "phone"]


async def test_a_message_reaches_every_device_of_one_user(
    manager: ConnectionManager,
) -> None:
    laptop, laptop_socket = connect("laptop")
    phone, phone_socket = connect("phone")
    await manager.aregister(laptop)
    await manager.aregister(phone)

    delivered = await manager.adeliver_locally(
        str(USER), RealtimeEnvelope(payload={"hello": True})
    )

    assert delivered == 2
    assert laptop_socket.frames()[0].payload == {"hello": True}
    assert phone_socket.frames()[0].payload == {"hello": True}


async def test_another_users_devices_are_left_alone(
    manager: ConnectionManager,
) -> None:
    mine, my_socket = connect("laptop")
    theirs, their_socket = connect("laptop", user=uuid4())
    await manager.aregister(mine)
    await manager.aregister(theirs)

    await manager.adeliver_locally(str(USER), RealtimeEnvelope())

    assert my_socket.sent
    assert their_socket.sent == []


async def test_a_reconnecting_device_replaces_its_old_socket(
    manager: ConnectionManager,
) -> None:
    """Keeping both would deliver everything twice to the same screen."""
    first, first_socket = connect("laptop")
    await manager.aregister(first)

    second, second_socket = connect("laptop")
    await manager.aregister(second)

    assert first_socket.closed is not None
    assert first_socket.closed[0] == RealtimeCloseCode.GOING_AWAY
    assert manager.devices_of(str(USER)) == ["laptop"]

    await manager.adeliver_locally(str(USER), RealtimeEnvelope())
    assert second_socket.sent


async def test_unregistering_forgets_only_that_socket(
    manager: ConnectionManager,
) -> None:
    laptop, _ = connect("laptop")
    phone, _ = connect("phone")
    await manager.aregister(laptop)
    await manager.aregister(phone)

    await manager.aunregister(laptop)
    assert manager.devices_of(str(USER)) == ["phone"]


async def test_a_replaced_socket_unregistering_late_is_harmless(
    manager: ConnectionManager,
) -> None:
    """The old socket's cleanup runs after its replacement took the slot."""
    first, _ = connect("laptop")
    await manager.aregister(first)
    second, _ = connect("laptop")
    await manager.aregister(second)

    await manager.aunregister(first)

    assert manager.devices_of(str(USER)) == ["laptop"]


async def test_a_closed_socket_is_skipped_rather_than_written_to(
    manager: ConnectionManager,
) -> None:
    connection, socket = connect("laptop")
    await manager.aregister(connection)
    await connection.aclose()

    assert await manager.adeliver_locally(str(USER), RealtimeEnvelope()) == 0


async def test_a_connected_user_is_online(manager: ConnectionManager) -> None:
    await manager.aregister(connect("laptop")[0])
    presence = await manager.apresence(str(USER))
    assert presence.online
    assert presence.last_seen is not None


async def test_an_unknown_user_is_offline(manager: ConnectionManager) -> None:
    presence = await manager.apresence(str(uuid4()))
    assert not presence.online
    assert presence.last_seen is None


async def test_presence_ends_when_the_last_device_goes(
    manager: ConnectionManager,
) -> None:
    laptop, _ = connect("laptop")
    phone, _ = connect("phone")
    await manager.aregister(laptop)
    await manager.aregister(phone)

    await manager.aunregister(laptop)
    assert (await manager.apresence(str(USER))).online

    await manager.aunregister(phone)
    assert not (await manager.apresence(str(USER))).online


async def test_presence_is_given_an_expiry(manager: ConnectionManager) -> None:
    """A replica killed without warning never cleans up after itself."""
    await manager.aregister(connect("laptop")[0])
    cache = manager.cache
    assert await cache.aget(str(USER), namespace=REALTIME_PRESENCE_NAMESPACE)


async def test_draining_tells_clients_before_closing_them(
    manager: ConnectionManager,
) -> None:
    """A close with no warning makes every client reconnect at once."""
    _, socket = connect("laptop")
    await manager.aregister(_)

    await manager.adrain("Rolling restart")

    kinds = [f.type for f in socket.frames()]
    assert RealtimeMessageType.CLOSING in kinds
    assert socket.closed is not None
    assert socket.closed[0] == RealtimeCloseCode.GOING_AWAY


async def test_draining_empties_the_registry(
    manager: ConnectionManager,
) -> None:
    await manager.aregister(connect("laptop")[0])
    await manager.adrain()
    assert manager.local_count() == 0
    assert not (await manager.apresence(str(USER))).online


async def test_a_draining_replica_refuses_new_sockets(
    manager: ConnectionManager,
) -> None:
    """Otherwise a client reconnects straight back into the process leaving."""
    await manager.adrain()
    assert manager.draining
    assert not await manager.aregister(connect("laptop")[0])


async def test_stopping_without_starting_is_harmless(
    manager: ConnectionManager,
) -> None:
    await manager.astop()


async def test_an_anonymous_socket_is_still_addressable(
    manager: ConnectionManager,
) -> None:
    """A verified token need not carry a user id, and still needs to be reachable."""
    socket = FakeSocket()
    anonymous = RealtimeConnection(
        socket,  # type: ignore[arg-type]
        ConnectionTicket(
            ticket="t",
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            subject="service-account",
            device_id="cli",
        ),
        limiter=InMemoryRateLimiter(),
    )
    await manager.aregister(anonymous)
    assert await manager.adeliver_locally("service-account", RealtimeEnvelope()) == 1


# Durable delivery


async def test_an_ephemeral_message_leaves_nothing_to_catch_up_on(
    manager: ConnectionManager,
) -> None:
    """A typing indicator nobody saw is not owed to anyone later."""
    user = str(uuid4())
    await manager.apublish_to_user(
        user, envelope(RealtimeMessageType.MESSAGE), ChannelDelivery.EPHEMERAL
    )
    assert await manager._inbox.asince(user, after_seq=0) == []


async def test_a_durable_message_is_recorded_before_it_is_sent(
    manager: ConnectionManager,
) -> None:
    """Recording second would lose exactly the messages durability is for."""
    user = str(uuid4())
    stored = await manager.apublish_to_user(
        user, envelope(RealtimeMessageType.MESSAGE), ChannelDelivery.DURABLE
    )
    assert stored.seq == 1
    assert len(await manager._inbox.asince(user, after_seq=0)) == 1


async def test_a_device_that_was_offline_is_sent_the_gap_on_reconnect(
    manager: ConnectionManager,
) -> None:
    user = uuid4()
    for _ in range(3):
        await manager.apublish_to_user(
            str(user), envelope(RealtimeMessageType.MESSAGE), ChannelDelivery.DURABLE
        )

    phone, socket = connect("phone", user)
    await manager.aregister(phone)
    sent = await manager.aresume(phone, last_seq=1)

    assert sent == 2
    assert [frame.seq for frame in socket.frames()] == [2, 3]


async def test_a_device_that_missed_nothing_is_sent_nothing(
    manager: ConnectionManager,
) -> None:
    user = uuid4()
    await manager.apublish_to_user(
        str(user), envelope(RealtimeMessageType.MESSAGE), ChannelDelivery.DURABLE
    )
    laptop, socket = connect("laptop", user)
    await manager.aregister(laptop)

    assert await manager.aresume(laptop, last_seq=1) == 0
    assert socket.frames() == []


async def test_acknowledging_moves_only_the_acknowledging_device(
    manager: ConnectionManager,
) -> None:
    user = uuid4()
    for _ in range(2):
        await manager.apublish_to_user(
            str(user), envelope(RealtimeMessageType.MESSAGE), ChannelDelivery.DURABLE
        )
    laptop, _ = connect("laptop", user)
    phone, _ = connect("phone", user)

    await manager.aacknowledge(laptop, 2)

    assert await manager.acursor_of(laptop) == 2
    assert await manager.acursor_of(phone) == 0


async def test_both_devices_are_owed_the_same_messages(
    manager: ConnectionManager,
) -> None:
    """One inbox, one cursor each: a laptop reading does not consume the phone's."""
    user = uuid4()
    for _ in range(2):
        await manager.apublish_to_user(
            str(user), envelope(RealtimeMessageType.MESSAGE), ChannelDelivery.DURABLE
        )
    laptop, laptop_socket = connect("laptop", user)
    phone, phone_socket = connect("phone", user)
    await manager.aregister(laptop)
    await manager.aregister(phone)

    await manager.aresume(laptop, last_seq=0)
    await manager.aacknowledge(laptop, 2)
    await manager.aresume(phone, last_seq=0)

    assert [f.seq for f in laptop_socket.frames()] == [1, 2]
    assert [f.seq for f in phone_socket.frames()] == [1, 2]


async def test_catching_up_arrives_in_the_order_it_was_published(
    manager: ConnectionManager,
) -> None:
    user = uuid4()
    for n in range(8):
        await manager.apublish_to_user(
            str(user),
            RealtimeEnvelope(type=RealtimeMessageType.MESSAGE, payload={"n": n}),
            ChannelDelivery.DURABLE,
        )
    laptop, socket = connect("laptop", user)
    await manager.aregister(laptop)
    await manager.aresume(laptop, last_seq=0)

    assert [f.payload["n"] for f in socket.frames()] == list(range(8))


async def test_an_ack_frame_advances_the_cursor_and_goes_no_further(
    manager: ConnectionManager,
) -> None:
    """Delivery frames are the manager's business, not the application's."""
    user = uuid4()
    await manager.apublish_to_user(
        str(user), envelope(RealtimeMessageType.MESSAGE), ChannelDelivery.DURABLE
    )
    laptop, _ = connect("laptop", user)

    handled = await manager.ahandle(
        laptop, RealtimeEnvelope(type=RealtimeMessageType.ACK, seq=1)
    )

    assert handled is None
    assert await manager.acursor_of(laptop) == 1


async def test_a_resume_frame_sends_the_gap_and_goes_no_further(
    manager: ConnectionManager,
) -> None:
    user = uuid4()
    for _ in range(2):
        await manager.apublish_to_user(
            str(user), envelope(RealtimeMessageType.MESSAGE), ChannelDelivery.DURABLE
        )
    laptop, socket = connect("laptop", user)
    await manager.aregister(laptop)

    handled = await manager.ahandle(
        laptop, RealtimeEnvelope(type=RealtimeMessageType.RESUME, seq=1)
    )

    assert handled is None
    assert [f.seq for f in socket.frames()] == [2]


async def test_a_resume_without_a_sequence_replays_everything(
    manager: ConnectionManager,
) -> None:
    """A client that has never connected holds nothing and says so."""
    user = uuid4()
    await manager.apublish_to_user(
        str(user), envelope(RealtimeMessageType.MESSAGE), ChannelDelivery.DURABLE
    )
    laptop, socket = connect("laptop", user)
    await manager.aregister(laptop)

    await manager.ahandle(laptop, RealtimeEnvelope(type=RealtimeMessageType.RESUME))

    assert [f.seq for f in socket.frames()] == [1]


async def test_an_application_frame_is_handed_back_untouched(
    manager: ConnectionManager,
) -> None:
    laptop, _ = connect("laptop")
    frame = RealtimeEnvelope(type=RealtimeMessageType.MESSAGE, payload={"a": 1})
    assert await manager.ahandle(laptop, frame) is frame


async def test_an_ack_without_a_sequence_is_ignored_rather_than_fatal(
    manager: ConnectionManager,
) -> None:
    """A malformed ack is one bad client, not a reason to drop the socket."""
    laptop, _ = connect("laptop", uuid4())
    assert (
        await manager.ahandle(laptop, RealtimeEnvelope(type=RealtimeMessageType.ACK))
        is None
    )
