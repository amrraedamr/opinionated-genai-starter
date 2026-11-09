"""What a user is owed when they were not there to receive it.

A device is offline far more often than it is connected: a phone in a tunnel,
a laptop asleep, a tab a browser froze to save battery. These pin the
difference between a channel that accepts that and one that does not, and the
part that makes several devices on one account work — one inbox, one cursor
each, so a laptop reading something does not consume it from the phone.
"""

import asyncio
from uuid import uuid4

import pytest

from base.models.definitions.realtime import ChannelDelivery, RealtimeMessageType
from base.models.view_models.realtime import RealtimeEnvelope
from base.services.cache.memory import InMemoryCacheService
from base.services.realtime.inbox import DurableInbox
from base.tools.tenancy import TenantContext


@pytest.fixture
def inbox() -> DurableInbox:
    return DurableInbox(InMemoryCacheService())


@pytest.fixture
def user() -> str:
    """A fresh account per test: the in-memory cache outlives any one of them."""
    return str(uuid4())


def message(text: str) -> RealtimeEnvelope:
    return RealtimeEnvelope(type=RealtimeMessageType.MESSAGE, payload={"text": text})


async def test_a_recorded_message_is_numbered_from_one(
    inbox: DurableInbox, user: str
) -> None:
    stored = await inbox.aappend(user, message("first"))
    assert stored.seq == 1


async def test_sequences_increase_without_gaps(inbox: DurableInbox, user: str) -> None:
    """A cursor is only meaningful if the next number is the next message."""
    seqs = [(await inbox.aappend(user, message(str(n)))).seq for n in range(5)]
    assert seqs == [1, 2, 3, 4, 5]


async def test_a_device_that_was_offline_receives_what_it_missed(
    inbox: DurableInbox, user: str
) -> None:
    for text in ("one", "two", "three"):
        await inbox.aappend(user, message(text))

    missed = await inbox.asince(user, after_seq=1)

    assert [m.payload["text"] for m in missed] == ["two", "three"]


async def test_catching_up_preserves_the_order_messages_were_sent_in(
    inbox: DurableInbox, user: str
) -> None:
    for n in range(10):
        await inbox.aappend(user, message(str(n)))

    missed = await inbox.asince(user, after_seq=0)

    assert [m.seq for m in missed] == list(range(1, 11))


async def test_a_device_that_is_up_to_date_receives_nothing(
    inbox: DurableInbox, user: str
) -> None:
    await inbox.aappend(user, message("only"))
    assert await inbox.asince(user, after_seq=1) == []


async def test_a_cursor_starts_at_nothing_read(inbox: DurableInbox, user: str) -> None:
    assert await inbox.acursor(user, "laptop") == 0


async def test_acknowledging_advances_only_the_device_that_acknowledged(
    inbox: DurableInbox, user: str
) -> None:
    """The point of a cursor per device.

    One inbox and one shared cursor would mean a laptop reading a message
    marks it read on the phone too, and the phone never shows it.
    """
    for text in ("one", "two"):
        await inbox.aappend(user, message(text))

    await inbox.aadvance(user, "laptop", 2)

    assert await inbox.acursor(user, "laptop") == 2
    assert await inbox.acursor(user, "phone") == 0


async def test_both_devices_are_owed_every_message(
    inbox: DurableInbox, user: str
) -> None:
    for text in ("one", "two"):
        await inbox.aappend(user, message(text))
    await inbox.aadvance(user, "laptop", 2)

    phone_missed = await inbox.asince(user, await inbox.acursor(user, "phone"))

    assert [m.payload["text"] for m in phone_missed] == ["one", "two"]


async def test_a_late_acknowledgement_does_not_rewind_a_cursor(
    inbox: DurableInbox, user: str
) -> None:
    """Acknowledgements race. A cursor is a high-water mark, not the last seen."""
    await inbox.aadvance(user, "laptop", 5)
    assert await inbox.aadvance(user, "laptop", 2) == 5


async def test_acknowledging_the_same_sequence_twice_is_harmless(
    inbox: DurableInbox, user: str
) -> None:
    await inbox.aadvance(user, "laptop", 3)
    assert await inbox.aadvance(user, "laptop", 3) == 3


async def test_one_users_inbox_is_not_another_users(
    inbox: DurableInbox, user: str
) -> None:
    other = str(uuid4())
    await inbox.aappend(user, message("private"))
    assert await inbox.asince(other, after_seq=0) == []


async def test_two_tenants_do_not_share_an_inbox(
    inbox: DurableInbox, user: str
) -> None:
    """Scoped by tenant as well as user, so a key cannot name the wrong one."""
    first, second = uuid4(), uuid4()

    with TenantContext.scope(first):
        await inbox.aappend(user, message("first tenant"))
    with TenantContext.scope(second):
        second_tenant_view = await inbox.asince(user, after_seq=0)
    with TenantContext.scope(first):
        first_tenant_view = await inbox.asince(user, after_seq=0)

    assert second_tenant_view == []
    assert [m.payload["text"] for m in first_tenant_view] == ["first tenant"]


async def test_concurrent_writers_never_share_a_sequence(
    inbox: DurableInbox, user: str
) -> None:
    """Two replicas publishing at once must not produce the same number.

    Two messages sharing a sequence makes a cursor ambiguous: a device
    resuming from it either misses one or receives one twice, and nothing in
    the protocol says which.
    """
    stored = await asyncio.gather(
        *(inbox.aappend(user, message(str(n))) for n in range(20))
    )

    assert sorted(m.seq for m in stored if m.seq is not None) == list(range(1, 21))


async def test_clearing_an_inbox_leaves_nothing_to_catch_up_on(
    inbox: DurableInbox, user: str
) -> None:
    await inbox.aappend(user, message("one"))
    await inbox.aclear(user)
    assert await inbox.asince(user, after_seq=0) == []


async def test_the_latest_sequence_of_an_empty_inbox_is_zero(
    inbox: DurableInbox, user: str
) -> None:
    assert await inbox.alatest_seq(user) == 0


async def test_the_latest_sequence_tracks_the_last_message(
    inbox: DurableInbox, user: str
) -> None:
    for text in ("one", "two", "three"):
        await inbox.aappend(user, message(text))
    assert await inbox.alatest_seq(user) == 3


async def test_an_ephemeral_channel_is_not_a_delivery_promise() -> None:
    """Named so the two are not interchangeable at a call site."""
    assert ChannelDelivery.EPHEMERAL != ChannelDelivery.DURABLE


async def test_a_device_outliving_the_inbox_is_resynced_not_silenced(
    inbox: DurableInbox, user: str
) -> None:
    """Retention expiring restarts numbering, and a stale cursor outranks it.

    Filtering on the old number would hand back nothing until the count
    climbed past it: silence that looks exactly like having nothing to say.
    """
    await inbox.aappend(user, message("after the reset"))

    caught_up = await inbox.asince(user, after_seq=99)

    assert [m.payload["text"] for m in caught_up] == ["after the reset"]


async def test_a_cleared_inbox_resyncs_rather_than_replays_from_a_stale_cursor(
    inbox: DurableInbox, user: str
) -> None:
    for text in ("one", "two", "three"):
        await inbox.aappend(user, message(text))
    await inbox.aclear(user)
    await inbox.aappend(user, message("new epoch"))

    assert [m.payload["text"] for m in await inbox.asince(user, after_seq=3)] == [
        "new epoch"
    ]
