"""The order `wrap` does things in, with the database stubbed out.

The claim has to be recorded before the handler runs and the outcome after it,
or a crash mid-handler is either lost or repeated. That sequencing is the
whole of `wrap`, so it is pinned here on its own; whether the claim actually
holds against a second process is a database question and lives in
`tests/integration/messaging`.
"""

from typing import Any

import pytest

from base.models.definitions.messaging import InboxState
from base.services.messaging import InboxProcessor
from contracts.events import Event


class Row:
    """Stands in for a service's inbox model."""


def build(monkeypatch: pytest.MonkeyPatch, *, claimed: bool = True) -> Any:
    """A processor whose claim and finish calls are recorded, not written."""
    monkeypatch.setattr(
        "base.services.messaging.inbox.SQLService", lambda *a, **k: object()
    )
    processor = InboxProcessor(Row)  # type: ignore[arg-type]
    trail: list[Any] = []

    async def claim(event: Event) -> bool:
        trail.append(("claim", event.meta.event_id))
        return claimed

    async def finish(
        event: Event, state: InboxState, error: Exception | None = None
    ) -> None:
        trail.append(("finish", state, type(error).__name__ if error else None))

    monkeypatch.setattr(processor, "_aclaim", claim)
    monkeypatch.setattr(processor, "_afinish", finish)
    return processor, trail


def event() -> Event:
    return Event.of("t", {"a": 1}, producer="p")


async def test_a_claimed_event_runs_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, trail = build(monkeypatch)
    ran: list[Event] = []

    async def handler(incoming: Event) -> None:
        ran.append(incoming)

    await processor.wrap(handler)(event())
    assert len(ran) == 1
    assert [step[0] for step in trail] == ["claim", "finish"]


async def test_an_unclaimed_event_never_reaches_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, trail = build(monkeypatch, claimed=False)
    ran: list[Event] = []

    async def handler(incoming: Event) -> None:
        ran.append(incoming)

    await processor.wrap(handler)(event())
    assert ran == []
    assert [step[0] for step in trail] == ["claim"]


async def test_a_successful_handler_is_recorded_as_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, trail = build(monkeypatch)

    async def handler(incoming: Event) -> None:
        return None

    await processor.wrap(handler)(event())
    assert trail[-1] == ("finish", InboxState.PROCESSED, None)


async def test_a_failing_handler_is_recorded_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, trail = build(monkeypatch)

    async def handler(incoming: Event) -> None:
        raise ValueError("handler failed")

    with pytest.raises(ValueError, match="handler failed"):
        await processor.wrap(handler)(event())
    assert trail[-1] == ("finish", InboxState.FAILED, "ValueError")


async def test_a_failing_handler_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The broker must see the failure, or it would ack and never redeliver."""
    processor, _ = build(monkeypatch)

    async def handler(incoming: Event) -> None:
        raise ValueError("handler failed")

    with pytest.raises(ValueError, match="handler failed"):
        await processor.wrap(handler)(event())


async def test_the_claim_is_recorded_before_the_handler_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, trail = build(monkeypatch)

    async def handler(incoming: Event) -> None:
        trail.append(("handler", None))

    await processor.wrap(handler)(event())
    assert [step[0] for step in trail] == ["claim", "handler", "finish"]


async def test_the_claim_names_the_event_being_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, trail = build(monkeypatch)
    delivered = event()

    async def handler(incoming: Event) -> None:
        return None

    await processor.wrap(handler)(delivered)
    assert trail[0] == ("claim", delivered.meta.event_id)


def test_wrapping_returns_a_different_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _ = build(monkeypatch)

    async def handler(incoming: Event) -> None:
        return None

    assert processor.wrap(handler) is not handler


def test_the_claim_timeout_is_the_caller_s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "base.services.messaging.inbox.SQLService", lambda *a, **k: object()
    )
    processor = InboxProcessor(Row, claim_timeout_seconds=42)  # type: ignore[arg-type]
    assert processor._claim_timeout.total_seconds() == 42


def test_the_claim_timeout_defaults_to_five_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "base.services.messaging.inbox.SQLService", lambda *a, **k: object()
    )
    assert InboxProcessor(Row)._claim_timeout.total_seconds() == 300  # type: ignore[arg-type]


def test_the_states_an_event_can_be_in() -> None:
    assert [state.value for state in InboxState] == [
        "processing",
        "processed",
        "failed",
    ]
