"""Connection tickets: the credential a websocket can actually be given.

A browser cannot put an `Authorization` header on a WebSocket, so something
else has to carry the identity. The usual answer is a token in the query
string, which writes a bearer credential into access logs and browser history.

These pin the properties that make a ticket a safer answer: it works once, it
expires, and it arrives somewhere nothing logs.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from base.models.definitions.realtime import (
    TICKET_SUBPROTOCOL_PREFIX,
    TICKET_TTL_SECONDS,
)
from base.models.definitions.tenancy import TenantId
from base.models.view_models.auth import Principal
from base.models.view_models.realtime import ConnectionTicket
from base.services.cache.memory import InMemoryCacheService
from base.services.realtime.tickets import TICKET_NAMESPACE, TicketService
from tests.conftest import MakeSettings


@pytest.fixture
def tickets(make_settings: MakeSettings) -> TicketService:
    make_settings()
    return TicketService(InMemoryCacheService())


def principal(**extra: Any) -> Principal:
    return Principal(subject="user-1", user_id=uuid4(), **extra)


async def test_a_ticket_can_be_redeemed_once(tickets: TicketService) -> None:
    issued = await tickets.aissue(principal())
    assert await tickets.aredeem(issued.ticket) is not None


async def test_a_ticket_cannot_be_redeemed_twice(tickets: TicketService) -> None:
    """Single use is the whole point: a replayed ticket is a second session."""
    issued = await tickets.aissue(principal())
    await tickets.aredeem(issued.ticket)
    assert await tickets.aredeem(issued.ticket) is None


async def test_an_unknown_ticket_is_refused(tickets: TicketService) -> None:
    assert await tickets.aredeem("never-issued") is None


async def test_no_ticket_at_all_is_refused(tickets: TicketService) -> None:
    assert await tickets.aredeem(None) is None
    assert await tickets.aredeem("") is None


async def test_an_expired_ticket_is_refused(tickets: TicketService) -> None:
    """Checked on the value as well as by cache expiry, which is approximate."""
    expired = ConnectionTicket(
        ticket="stale",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
        subject="user-1",
        device_id="laptop",
    )
    await tickets.cache.aset(
        "stale", namespace=TICKET_NAMESPACE, value=expired.jsonify()
    )
    assert await tickets.aredeem("stale") is None


async def test_an_expired_ticket_is_still_consumed(tickets: TicketService) -> None:
    """Refusing without spending it would leave a stale row to retry against."""
    expired = ConnectionTicket(
        ticket="stale",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
        subject="user-1",
        device_id="laptop",
    )
    await tickets.cache.aset(
        "stale", namespace=TICKET_NAMESPACE, value=expired.jsonify()
    )
    await tickets.aredeem("stale")
    assert await tickets.cache.aget("stale", namespace=TICKET_NAMESPACE) is None


async def test_a_ticket_expires_in_well_under_a_minute(
    tickets: TicketService,
) -> None:
    """Long enough to open a socket, short enough that a leak is worthless."""
    issued = await tickets.aissue(principal())
    remaining = (issued.expires_at - datetime.now(UTC)).total_seconds()
    assert 0 < remaining <= TICKET_TTL_SECONDS


async def test_a_ticket_carries_the_identity_the_socket_will_act_as(
    tickets: TicketService,
) -> None:
    """The handshake cannot verify a token, so the ticket has to hold this."""
    caller = principal(tenant_id=TenantId("acme"))
    issued = await tickets.aissue(caller)

    redeemed = await tickets.aredeem(issued.ticket)
    assert redeemed is not None
    assert redeemed.subject == caller.subject
    assert redeemed.user_id == caller.user_id
    assert redeemed.tenant_id == "acme"


async def test_the_reply_does_not_restate_who_the_caller_is(
    tickets: TicketService,
) -> None:
    """The client already knows; a copy in the reply is one more thing to leak."""
    issued = await tickets.aissue(principal(tenant_id=TenantId("acme")))
    dumped = issued.jsonify()
    assert "acme" not in dumped
    assert "user-1" not in dumped


async def test_two_tickets_are_never_the_same(tickets: TicketService) -> None:
    first = await tickets.aissue(principal())
    second = await tickets.aissue(principal())
    assert first.ticket != second.ticket


async def test_the_reply_says_exactly_what_to_send(
    tickets: TicketService,
) -> None:
    """So a client never has to guess how the prefix is spelled."""
    issued = await tickets.aissue(principal())
    assert issued.subprotocol == f"{TICKET_SUBPROTOCOL_PREFIX}{issued.ticket}"


def test_a_ticket_is_read_from_the_subprotocol_header() -> None:
    assert (
        TicketService.from_subprotocols([f"{TICKET_SUBPROTOCOL_PREFIX}abc123"])
        == "abc123"
    )


def test_an_offer_without_a_ticket_yields_nothing() -> None:
    assert TicketService.from_subprotocols(["json", "graphql-ws"]) is None
    assert TicketService.from_subprotocols([]) is None


def test_an_empty_ticket_is_not_a_ticket() -> None:
    """`ticket.` with nothing after it must not read as a valid empty one."""
    assert TicketService.from_subprotocols([TICKET_SUBPROTOCOL_PREFIX]) is None


def test_a_ticket_is_picked_out_from_among_other_protocols() -> None:
    offered = ["json", f"{TICKET_SUBPROTOCOL_PREFIX}abc", "graphql-ws"]
    assert TicketService.from_subprotocols(offered) == "abc"


async def test_a_redeemed_ticket_survives_the_cache_round_trip(
    tickets: TicketService,
) -> None:
    """It is stored as JSON, so the expiry must come back a time, not a string."""
    issued = await tickets.aissue(principal())
    redeemed = await tickets.aredeem(issued.ticket)
    assert redeemed is not None
    assert isinstance(redeemed.expires_at, datetime)
    assert not redeemed.expired


# One user, several clients


async def test_the_client_chooses_its_own_device_identity(
    tickets: TicketService,
) -> None:
    """A server-minted one would change on every reconnect, which is useless."""
    caller = principal()
    issued = await tickets.aissue(caller, device_id="laptop")
    redeemed = await tickets.aredeem(issued.ticket)
    assert redeemed is not None
    assert redeemed.device_id == "laptop"


async def test_a_device_identity_survives_reconnecting(
    tickets: TicketService,
) -> None:
    """The same laptop coming back is the same device, with the same cursor."""
    caller = principal()
    first = await tickets.aissue(caller, device_id="laptop")
    second = await tickets.aissue(caller, device_id="laptop")

    assert first.device_id == second.device_id == "laptop"
    assert first.ticket != second.ticket


async def test_two_clients_of_one_user_are_told_apart(
    tickets: TicketService,
) -> None:
    """A laptop and a phone, same person: one user, two delivery targets."""
    caller = principal()
    laptop = await tickets.aredeem(
        (await tickets.aissue(caller, device_id="laptop")).ticket
    )
    phone = await tickets.aredeem(
        (await tickets.aissue(caller, device_id="phone")).ticket
    )

    assert laptop is not None and phone is not None
    assert laptop.user_id == phone.user_id
    assert laptop.device_id != phone.device_id


async def test_a_client_that_names_no_device_is_given_one(
    tickets: TicketService,
) -> None:
    """A first connection has nothing to send, and must still work."""
    issued = await tickets.aissue(principal())
    assert issued.device_id


async def test_the_issued_device_identity_is_handed_back_to_keep(
    tickets: TicketService,
) -> None:
    """Otherwise a client cannot resume, because it never learned its own name."""
    issued = await tickets.aissue(principal())
    redeemed = await tickets.aredeem(issued.ticket)
    assert redeemed is not None
    assert redeemed.device_id == issued.device_id


async def test_each_socket_gets_its_own_connection_identity(
    tickets: TicketService,
) -> None:
    """A reconnect is the same device and a different connection.

    The registry has to drop the old socket without forgetting where that
    device had read to.
    """
    caller = principal()
    first = await tickets.aredeem(
        (await tickets.aissue(caller, device_id="laptop")).ticket
    )
    second = await tickets.aredeem(
        (await tickets.aissue(caller, device_id="laptop")).ticket
    )

    assert first is not None and second is not None
    assert first.device_id == second.device_id
    assert first.connection_id != second.connection_id
