"""The three bounds a websocket does not get for free, and the handshake.

Every HTTP request passes a size limit and a rate limiter. A socket passes
neither — both middlewares return early on anything that is not HTTP — so it
is admitted once and then sends what it likes for as long as it likes.

Heartbeats cover the other half: a peer that vanished without a close frame is
indistinguishable from an idle one, and nothing notices until something tries
to write, which for a subscription may be never.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from base.models.definitions.realtime import (
    TICKET_SUBPROTOCOL_PREFIX,
    RealtimeCloseCode,
    RealtimeMessageType,
)
from base.models.view_models.realtime import ConnectionTicket, RealtimeEnvelope
from base.services.cache.memory import InMemoryCacheService
from base.services.configuration.mixins import RateLimitSettingsMixin
from base.services.rate_limiting.memory import InMemoryRateLimiter
from base.services.realtime.connection import RealtimeConnection
from base.services.realtime.handshake import accept_connection
from base.services.realtime.tickets import TicketService


class FakeSocket:
    """Records what was sent, and hands back what a test queued."""

    def __init__(self, incoming: list[str] | None = None, **scope: Any) -> None:
        self.incoming = list(incoming or [])
        self.sent: list[str] = []
        self.scope: dict[str, Any] = scope
        self.accepted_with: Any = "not-accepted"
        self.closed: tuple[int, str] | None = None

    async def receive_text(self) -> str:
        if not self.incoming:
            raise AssertionError("the connection asked for more than was queued")
        return self.incoming.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def accept(self, subprotocol: Any = None) -> None:
        self.accepted_with = subprotocol

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)

    def frames(self) -> list[RealtimeEnvelope]:
        return [RealtimeEnvelope.model_validate_json(raw) for raw in self.sent]


def ticket() -> ConnectionTicket:
    return ConnectionTicket(
        ticket="t",
        expires_at=datetime.now(UTC) + timedelta(seconds=30),
        subject="user-1",
        user_id=uuid4(),
        device_id="laptop",
    )


@pytest.fixture
def limited(make_settings: Any) -> Any:
    def _build(per_minute: int = 100) -> InMemoryRateLimiter:
        make_settings(
            RateLimitSettingsMixin, rate_limit_ws_messages_per_minute=per_minute
        )
        return InMemoryRateLimiter()

    return _build


def connection(socket: FakeSocket, limiter: Any, **kwargs: Any) -> RealtimeConnection:
    return RealtimeConnection(socket, ticket(), limiter=limiter, **kwargs)


async def test_an_ordinary_frame_is_delivered(limited: Any) -> None:
    socket = FakeSocket([RealtimeEnvelope(payload={"a": 1}).jsonify()])
    received = await connection(socket, limited()).areceive()
    assert received is not None
    assert received.payload == {"a": 1}


async def test_an_oversize_frame_is_refused(limited: Any) -> None:
    """A socket bypasses the request size limit entirely."""
    huge = RealtimeEnvelope(payload={"a": "x" * 500}).jsonify()
    socket = FakeSocket([huge])

    assert await connection(socket, limited(), max_message_bytes=64).areceive() is None
    errors = [f for f in socket.frames() if f.type == RealtimeMessageType.ERROR]
    assert errors and "too large" in errors[0].payload["reason"].lower()


async def test_an_oversize_frame_does_not_close_the_socket(limited: Any) -> None:
    """One bad client loop is not worth a reconnect storm."""
    socket = FakeSocket([RealtimeEnvelope(payload={"a": "x" * 500}).jsonify()])
    live = connection(socket, limited(), max_message_bytes=64)
    await live.areceive()
    assert socket.closed is None
    assert not live.closed


async def test_a_frame_over_the_rate_limit_is_refused(limited: Any) -> None:
    """A socket bypasses the traffic middleware, so this is the only bound."""
    frame = RealtimeEnvelope(payload={}).jsonify()
    socket = FakeSocket([frame, frame])
    live = connection(socket, limited(per_minute=1))

    assert await live.areceive() is not None
    assert await live.areceive() is None
    errors = [f for f in socket.frames() if f.type == RealtimeMessageType.ERROR]
    assert errors and "too many" in errors[0].payload["reason"].lower()


async def test_the_budget_is_per_user_not_per_connection(limited: Any) -> None:
    """Ten sockets must not be ten budgets, or the limit means nothing."""
    limiter = limited(per_minute=1)
    shared = ticket()
    first = RealtimeConnection(
        FakeSocket([RealtimeEnvelope().jsonify()]), shared, limiter=limiter
    )
    second_socket = FakeSocket([RealtimeEnvelope().jsonify()])
    second = RealtimeConnection(second_socket, shared, limiter=limiter)

    assert await first.areceive() is not None
    assert await second.areceive() is None


async def test_a_malformed_frame_is_refused(limited: Any) -> None:
    socket = FakeSocket(["{not json"])
    assert await connection(socket, limited()).areceive() is None
    errors = [f for f in socket.frames() if f.type == RealtimeMessageType.ERROR]
    assert errors and "malformed" in errors[0].payload["reason"].lower()


async def test_a_frame_from_an_unknown_version_is_refused(limited: Any) -> None:
    """A client built against a newer shape must be told, not guessed at."""
    socket = FakeSocket(['{"type":"message","schemaVersion":99,"payload":{}}'])
    assert await connection(socket, limited()).areceive() is None


async def test_a_ping_is_answered_without_reaching_the_caller(
    limited: Any,
) -> None:
    socket = FakeSocket([RealtimeEnvelope.ping().jsonify()])
    assert await connection(socket, limited()).areceive() is None
    assert [f.type for f in socket.frames()] == [RealtimeMessageType.PONG]


async def test_a_pong_is_swallowed(limited: Any) -> None:
    """It is liveness, not application traffic."""
    socket = FakeSocket([RealtimeEnvelope.pong().jsonify()])
    assert await connection(socket, limited()).areceive() is None
    assert socket.sent == []


async def test_a_quiet_peer_is_eventually_considered_dead(limited: Any) -> None:
    """Otherwise a vanished client holds its connection open forever."""
    live = connection(FakeSocket(), limited(), heartbeat_timeout_seconds=0)
    assert live.stale


async def test_a_peer_that_just_spoke_is_not_stale(limited: Any) -> None:
    socket = FakeSocket([RealtimeEnvelope().jsonify()])
    live = connection(socket, limited(), heartbeat_timeout_seconds=60)
    await live.areceive()
    assert not live.stale


async def test_a_dead_connection_is_closed_with_a_timeout_code(
    limited: Any,
) -> None:
    socket = FakeSocket()
    live = connection(socket, limited(), heartbeat_timeout_seconds=0)
    await live.aclose(RealtimeCloseCode.TIMED_OUT, "No response to heartbeat")
    assert socket.closed == (RealtimeCloseCode.TIMED_OUT, "No response to heartbeat")


async def test_closing_twice_is_harmless(limited: Any) -> None:
    """Most implementations treat a second close as an error."""
    socket = FakeSocket()
    live = connection(socket, limited())
    await live.aclose()
    socket.closed = None
    await live.aclose()
    assert socket.closed is None


async def test_a_failed_send_closes_rather_than_raising(limited: Any) -> None:
    """A peer that vanished mid-write is ordinary, not exceptional."""

    class Broken(FakeSocket):
        async def send_text(self, text: str) -> None:
            raise ConnectionResetError("gone")

    live = connection(Broken(), limited())
    await live.asend(RealtimeEnvelope())
    assert live.closed


async def test_sending_after_close_is_a_no_op(limited: Any) -> None:
    socket = FakeSocket()
    live = connection(socket, limited())
    await live.aclose()
    await live.asend(RealtimeEnvelope())
    assert socket.sent == []


# Handshake


@pytest.fixture
def issuing(make_settings: Any) -> TicketService:
    make_settings()
    service = TicketService(InMemoryCacheService())
    return service


async def test_a_socket_with_a_valid_ticket_is_accepted(
    issuing: TicketService,
) -> None:
    from base.models.view_models.auth import Principal

    issued = await issuing.aissue(Principal(subject="user-1"))
    socket = FakeSocket(subprotocols=[issued.subprotocol])

    accepted = await accept_connection(socket)
    assert accepted is not None
    assert socket.closed is None


async def test_the_accepted_subprotocol_is_echoed_back(
    issuing: TicketService,
) -> None:
    """A client that offered one and is answered with none must fail the socket."""
    from base.models.view_models.auth import Principal

    issued = await issuing.aissue(Principal(subject="user-1"))
    socket = FakeSocket(subprotocols=[issued.subprotocol])

    await accept_connection(socket)
    assert socket.accepted_with == f"{TICKET_SUBPROTOCOL_PREFIX}{issued.ticket}"


async def test_a_socket_with_no_ticket_is_refused(issuing: TicketService) -> None:
    del issuing
    socket = FakeSocket(subprotocols=[])
    assert await accept_connection(socket) is None
    assert socket.closed is not None
    assert socket.closed[0] == RealtimeCloseCode.UNAUTHORIZED


async def test_a_refused_socket_is_never_accepted_first(
    issuing: TicketService,
) -> None:
    """Accepting then closing tells a client it is connected when it is not."""
    del issuing
    socket = FakeSocket(subprotocols=["ticket.forged"])
    await accept_connection(socket)
    assert socket.accepted_with == "not-accepted"


async def test_a_ticket_cannot_open_two_sockets(issuing: TicketService) -> None:
    from base.models.view_models.auth import Principal

    issued = await issuing.aissue(Principal(subject="user-1"))

    assert await accept_connection(FakeSocket(subprotocols=[issued.subprotocol]))
    second = FakeSocket(subprotocols=[issued.subprotocol])
    assert await accept_connection(second) is None
    assert second.closed is not None
