"""The dispatcher's polling loop, with the database stubbed out.

`_run` is the part that has to survive a bad night: a failed poll must not
kill the task, because a dead dispatcher stops publishing silently and the
outbox simply fills up. The loop is driven here through a stubbed
`adispatch_once` so a failure costs nothing and no row is needed.
"""

import asyncio
from typing import Any

from base.services.messaging import OutboxDispatcher
from base.services.messaging.outbox import MAX_BACKOFF_SECONDS
from contracts.events import Event


class CollectingBroker:
    """A broker that records what it was asked to publish."""

    def __init__(self) -> None:
        self.published: list[Event] = []

    async def apublish(self, event: Event) -> None:
        self.published.append(event)


class Row:
    """Stands in for a service's outbox model."""


def build(**kwargs: Any) -> OutboxDispatcher:
    return OutboxDispatcher(CollectingBroker(), Row, **kwargs)  # type: ignore[arg-type]


async def test_starting_creates_one_task() -> None:
    dispatcher = build(poll_seconds=0.01)
    dispatcher.adispatch_once = _returns(0)  # type: ignore[method-assign]
    await dispatcher.astart()
    assert dispatcher._task is not None
    await dispatcher.astop()


async def test_starting_twice_runs_one_task() -> None:
    dispatcher = build(poll_seconds=0.01)
    dispatcher.adispatch_once = _returns(0)  # type: ignore[method-assign]
    await dispatcher.astart()
    task = dispatcher._task
    await dispatcher.astart()
    assert dispatcher._task is task
    await dispatcher.astop()


async def test_the_task_is_named_for_the_logs() -> None:
    dispatcher = build(poll_seconds=0.01)
    dispatcher.adispatch_once = _returns(0)  # type: ignore[method-assign]
    await dispatcher.astart()
    assert dispatcher._task is not None
    assert dispatcher._task.get_name() == "outbox-dispatcher"
    await dispatcher.astop()


async def test_stopping_clears_the_task() -> None:
    dispatcher = build(poll_seconds=0.01)
    dispatcher.adispatch_once = _returns(0)  # type: ignore[method-assign]
    await dispatcher.astart()
    await dispatcher.astop()
    assert dispatcher._task is None


async def test_stopping_twice_is_accepted() -> None:
    dispatcher = build(poll_seconds=0.01)
    dispatcher.adispatch_once = _returns(0)  # type: ignore[method-assign]
    await dispatcher.astart()
    await dispatcher.astop()
    await dispatcher.astop()
    assert dispatcher._task is None


async def test_stopping_without_starting_is_accepted() -> None:
    dispatcher = build()
    await dispatcher.astop()
    assert dispatcher._task is None


async def test_a_stopped_dispatcher_starts_again() -> None:
    dispatcher = build(poll_seconds=0.01)
    calls = _counting(1)
    dispatcher.adispatch_once = calls  # type: ignore[method-assign]
    await dispatcher.astart()
    await dispatcher.astop()
    first_round = calls.count
    await dispatcher.astart()
    await asyncio.sleep(0.05)
    await dispatcher.astop()
    assert calls.count > first_round


async def test_the_loop_keeps_polling() -> None:
    dispatcher = build(poll_seconds=0.001)
    calls = _counting(0)
    dispatcher.adispatch_once = calls  # type: ignore[method-assign]
    await dispatcher.astart()
    await asyncio.sleep(0.1)
    await dispatcher.astop()
    assert calls.count > 1


async def test_a_failed_poll_does_not_kill_the_loop() -> None:
    """A dead task would stop publishing silently and forever."""
    dispatcher = build(poll_seconds=0.001)
    calls = _raising()
    dispatcher.adispatch_once = calls  # type: ignore[method-assign]
    await dispatcher.astart()
    await asyncio.sleep(0.1)
    assert dispatcher._task is not None
    assert not dispatcher._task.done()
    await dispatcher.astop()
    assert calls.count > 1


async def test_a_busy_dispatcher_does_not_wait_between_batches() -> None:
    """A full batch means more rows are ready, so it comes straight back."""
    slow = build(poll_seconds=5)
    busy = _counting(1)
    slow.adispatch_once = busy  # type: ignore[method-assign]
    await slow.astart()
    await asyncio.sleep(0.05)
    await slow.astop()
    assert busy.count > 5


async def test_an_idle_dispatcher_waits_the_poll_interval() -> None:
    dispatcher = build(poll_seconds=5)
    idle = _counting(0)
    dispatcher.adispatch_once = idle  # type: ignore[method-assign]
    await dispatcher.astart()
    await asyncio.sleep(0.05)
    await dispatcher.astop()
    assert idle.count == 1


async def test_stopping_wakes_an_idle_dispatcher_rather_than_waiting() -> None:
    dispatcher = build(poll_seconds=30)
    dispatcher.adispatch_once = _returns(0)  # type: ignore[method-assign]
    await dispatcher.astart()
    await asyncio.sleep(0.01)
    await asyncio.wait_for(dispatcher.astop(), timeout=2)
    assert dispatcher._task is None


def test_the_batch_size_and_poll_interval_are_the_caller_s() -> None:
    dispatcher = build(poll_seconds=2.5, batch_size=7)
    assert dispatcher._poll_seconds == 2.5
    assert dispatcher._batch_size == 7


def test_the_backoff_is_capped() -> None:
    """Otherwise a long outage would park a row for years."""
    assert MAX_BACKOFF_SECONDS == 60


def _returns(value: int) -> Any:
    async def _dispatch() -> int:
        await asyncio.sleep(0)
        return value

    return _dispatch


class _Counting:
    def __init__(self, value: int) -> None:
        self.value = value
        self.count = 0

    async def __call__(self) -> int:
        self.count += 1
        await asyncio.sleep(0)
        return self.value


class _Raising:
    def __init__(self) -> None:
        self.count = 0

    async def __call__(self) -> int:
        self.count += 1
        await asyncio.sleep(0)
        raise RuntimeError("the database is down")


def _counting(value: int) -> _Counting:
    return _Counting(value)


def _raising() -> _Raising:
    return _Raising()
