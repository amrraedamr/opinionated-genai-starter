"""`correlation.py` and `attempt_trail.py`: per-task state that must not leak.

Both are ContextVars because one process serves many requests at once. A value
set in one task bleeding into another would attach the wrong correlation id to
a log line, or the wrong attempt count to a usage record.
"""

import asyncio
import contextvars
from uuid import UUID, uuid4

from base.tools.attempt_trail import (
    AttemptTrail,
    mark_attempt,
    mark_attempt_failure,
    start_attempt_trail,
    take_attempt_trail,
)
from base.tools.correlation import (
    CORRELATION_HEADER,
    TRACEPARENT_HEADER,
    adopt_correlation_id,
    correlation_id,
    current_correlation_id,
    parse_correlation_id,
)


def test_the_header_names_are_the_ones_on_the_wire() -> None:
    assert CORRELATION_HEADER == "X-Correlation-ID"
    assert TRACEPARENT_HEADER == "traceparent"


def test_a_well_formed_uuid_header_is_adopted_as_sent() -> None:
    value = uuid4()
    assert parse_correlation_id(str(value)) == value


def test_a_traceparent_contributes_its_trace_id() -> None:
    trace = "4bf92f3577b34da6a3ce929d0e0e4736"
    header = f"00-{trace}-00f067aa0ba902b7-01"
    assert parse_correlation_id(header) == UUID(trace)


def test_a_traceparent_with_surrounding_space_is_still_read() -> None:
    header = "  00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01  "
    assert parse_correlation_id(header) == UUID("4bf92f3577b34da6a3ce929d0e0e4736")


def test_a_malformed_traceparent_is_refused() -> None:
    assert parse_correlation_id("00-tooshort-00f067aa0ba902b7-01") is None
    assert parse_correlation_id("not a header at all") is None


def test_an_absent_header_yields_nothing_to_adopt() -> None:
    assert parse_correlation_id(None) is None
    assert parse_correlation_id("") is None


def test_adopting_an_inbound_id_sets_it_for_this_context() -> None:
    value = uuid4()
    assert adopt_correlation_id(str(value)) == value
    assert current_correlation_id() == value


def test_the_correlation_header_wins_over_the_traceparent() -> None:
    value = uuid4()
    adopted = adopt_correlation_id(
        str(value), "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    )
    assert adopted == value


def test_a_traceparent_is_used_when_no_correlation_header_arrived() -> None:
    trace = "4bf92f3577b34da6a3ce929d0e0e4736"
    assert adopt_correlation_id(None, f"00-{trace}-00f067aa0ba902b7-01") == UUID(trace)


def test_a_request_with_no_usable_header_starts_a_new_id() -> None:
    first = adopt_correlation_id("garbage")
    second = adopt_correlation_id("garbage")
    assert first != second


def test_reading_an_unset_context_starts_an_id_rather_than_failing() -> None:
    def fresh() -> UUID:
        return current_correlation_id()

    value = contextvars.copy_context().run(fresh)
    assert isinstance(value, UUID)


def test_reading_twice_in_one_context_gives_the_same_id() -> None:
    def twice() -> tuple[UUID, UUID]:
        return current_correlation_id(), current_correlation_id()

    first, second = contextvars.copy_context().run(twice)
    assert first == second


async def test_concurrent_tasks_each_keep_their_own_correlation_id() -> None:
    async def handle(value: UUID) -> UUID:
        adopt_correlation_id(str(value))
        await asyncio.sleep(0)
        return current_correlation_id()

    wanted = [uuid4() for _ in range(5)]
    assert await asyncio.gather(*(handle(v) for v in wanted)) == wanted


async def test_a_task_does_not_overwrite_its_parents_correlation_id() -> None:
    parent = uuid4()
    adopt_correlation_id(str(parent))

    async def child() -> None:
        adopt_correlation_id(str(uuid4()))

    await asyncio.create_task(child())
    assert current_correlation_id() == parent


def test_a_fresh_trail_starts_at_no_attempts_and_no_errors() -> None:
    start_attempt_trail()
    assert take_attempt_trail() == AttemptTrail(attempts=0, errors=())


def test_each_attempt_is_counted() -> None:
    start_attempt_trail()
    mark_attempt()
    mark_attempt()
    assert take_attempt_trail().attempts == 2


def test_a_failed_attempt_records_its_type_and_its_message() -> None:
    start_attempt_trail()
    mark_attempt()
    mark_attempt_failure(ValueError("bad input"))
    trail = take_attempt_trail()
    assert trail.errors == ("ValueError: bad input",)


def test_failures_are_kept_in_the_order_they_happened() -> None:
    start_attempt_trail()
    mark_attempt_failure(TimeoutError("first"))
    mark_attempt_failure(ConnectionError("second"))
    assert take_attempt_trail().errors == (
        "TimeoutError: first",
        "ConnectionError: second",
    )


def test_reading_the_trail_clears_it_so_the_next_call_starts_clean() -> None:
    start_attempt_trail()
    mark_attempt()
    take_attempt_trail()
    assert take_attempt_trail() == AttemptTrail()


def test_a_trail_can_be_read_without_ever_being_started() -> None:
    def unstarted() -> AttemptTrail:
        return take_attempt_trail()

    assert contextvars.copy_context().run(unstarted) == AttemptTrail()


def test_starting_a_trail_discards_what_the_last_call_left() -> None:
    start_attempt_trail()
    mark_attempt()
    mark_attempt_failure(ValueError("old"))
    start_attempt_trail()
    assert take_attempt_trail() == AttemptTrail()


async def test_concurrent_calls_each_keep_their_own_attempt_trail() -> None:
    async def call(attempts: int) -> AttemptTrail:
        start_attempt_trail()
        for index in range(attempts):
            mark_attempt()
            await asyncio.sleep(0)
            mark_attempt_failure(TimeoutError(f"try {index}"))
        return take_attempt_trail()

    results = await asyncio.gather(call(1), call(2), call(3))
    assert [trail.attempts for trail in results] == [1, 2, 3]
    assert [len(trail.errors) for trail in results] == [1, 2, 3]


def test_the_trail_is_an_immutable_record_of_one_call() -> None:
    trail = AttemptTrail(attempts=2, errors=("a",))
    assert trail._replace(attempts=3) == AttemptTrail(attempts=3, errors=("a",))
    assert trail.attempts == 2


def test_the_correlation_var_is_the_one_the_logger_reads() -> None:
    value = uuid4()
    correlation_id.set(value)
    assert current_correlation_id() == value
