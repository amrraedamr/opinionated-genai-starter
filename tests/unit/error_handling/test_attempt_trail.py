"""What one outbound call went through, and why it cannot leak into the next.

The trail is a ContextVar rather than handler state because one handler serves
many concurrent calls. Two things matter: reading it clears it, so a later
record cannot inherit an earlier call's history, and two tasks running at once
never see each other's attempts.
"""

from __future__ import annotations

import asyncio
import contextvars

import pytest

from base.tools.attempt_trail import (
    AttemptTrail,
    mark_attempt,
    mark_attempt_failure,
    start_attempt_trail,
    take_attempt_trail,
    trail,
)


def test_a_fresh_trail_has_nothing_in_it() -> None:
    start_attempt_trail()
    assert take_attempt_trail() == AttemptTrail(attempts=0, errors=())


def test_an_unstarted_trail_reads_as_empty_rather_than_failing() -> None:
    """A fresh context has no value at all, and `take` must not raise there."""

    def probe() -> tuple[bool, AttemptTrail]:
        try:
            trail.get()
        except LookupError:
            return True, take_attempt_trail()
        return False, take_attempt_trail()

    unset, recorded = contextvars.Context().run(probe)
    assert unset
    assert recorded == AttemptTrail()


def test_each_attempt_is_counted() -> None:
    start_attempt_trail()
    for _ in range(3):
        mark_attempt()
    assert take_attempt_trail().attempts == 3


def test_a_failure_is_recorded_with_its_type_and_message() -> None:
    start_attempt_trail()
    mark_attempt()
    mark_attempt_failure(ValueError("could not parse"))
    assert take_attempt_trail().errors == ("ValueError: could not parse",)


def test_failures_are_kept_in_the_order_they_happened() -> None:
    start_attempt_trail()
    mark_attempt_failure(TimeoutError("first"))
    mark_attempt_failure(ConnectionError("second"))
    mark_attempt_failure(RuntimeError("third"))
    assert take_attempt_trail().errors == (
        "TimeoutError: first",
        "ConnectionError: second",
        "RuntimeError: third",
    )


def test_reading_the_trail_clears_it() -> None:
    start_attempt_trail()
    mark_attempt()
    mark_attempt_failure(ValueError("boom"))
    assert take_attempt_trail().attempts == 1
    assert take_attempt_trail() == AttemptTrail()


def test_starting_a_call_discards_what_the_last_one_left() -> None:
    start_attempt_trail()
    mark_attempt()
    mark_attempt_failure(ValueError("stale"))
    start_attempt_trail()
    mark_attempt()
    assert take_attempt_trail() == AttemptTrail(attempts=1, errors=())


def test_a_trail_is_an_immutable_record() -> None:
    start_attempt_trail()
    mark_attempt()
    recorded = take_attempt_trail()
    with pytest.raises(AttributeError):
        recorded.attempts = 5  # type: ignore[misc]
    assert isinstance(recorded.errors, tuple)


def test_an_attempt_count_and_a_failure_count_can_differ() -> None:
    start_attempt_trail()
    mark_attempt()
    mark_attempt()
    mark_attempt_failure(TimeoutError("only the first failed"))
    recorded = take_attempt_trail()
    assert (recorded.attempts, len(recorded.errors)) == (2, 1)


def test_two_concurrent_tasks_never_see_each_others_attempts() -> None:
    async def call(name: str, attempts: int) -> tuple[str, AttemptTrail]:
        start_attempt_trail()
        for index in range(attempts):
            mark_attempt()
            await asyncio.sleep(0)
            mark_attempt_failure(RuntimeError(f"{name}-{index}"))
        return name, take_attempt_trail()

    async def main() -> list[tuple[str, AttemptTrail]]:
        return await asyncio.gather(call("a", 1), call("b", 3))

    results = dict(asyncio.run(main()))
    assert results["a"] == AttemptTrail(attempts=1, errors=("RuntimeError: a-0",))
    assert results["b"].attempts == 3
    assert results["b"].errors == (
        "RuntimeError: b-0",
        "RuntimeError: b-1",
        "RuntimeError: b-2",
    )


def test_a_task_cannot_clear_the_trail_of_the_task_that_spawned_it() -> None:
    async def inner() -> None:
        start_attempt_trail()
        mark_attempt()
        take_attempt_trail()

    async def main() -> AttemptTrail:
        start_attempt_trail()
        mark_attempt()
        mark_attempt()
        await asyncio.create_task(inner())
        return take_attempt_trail()

    assert asyncio.run(main()).attempts == 2


def test_an_awaited_coroutine_shares_the_callers_trail() -> None:
    """Only a task gets its own context; a plain await does not."""

    async def inner() -> None:
        mark_attempt()

    async def main() -> AttemptTrail:
        start_attempt_trail()
        mark_attempt()
        await inner()
        return take_attempt_trail()

    assert asyncio.run(main()).attempts == 2


def test_a_copied_context_carries_the_trail_it_was_copied_with() -> None:
    start_attempt_trail()
    mark_attempt()
    captured: list[int] = []
    contextvars.copy_context().run(lambda: captured.append(trail.get().attempts))
    assert captured == [1]
    assert take_attempt_trail().attempts == 1
