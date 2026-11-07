"""Background tasks are named and registered so a distributed backend can run them.

The in-memory reference backend keeps the same API, which is what makes the
switch to a queue a configuration change rather than a rewrite. Its one extra
is `spawn_coroutine`, for a closure over request scope that cannot be named.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator

import pytest

from base.models.definitions.background_tasks import BackgroundTaskBackend, TaskStatus
from base.services.background_tasks.in_memory import InMemoryBackgroundTaskService
from base.services.common.meta import Singleton


@pytest.fixture(autouse=True)
def _fresh_service() -> Iterator[None]:
    Singleton.purge()
    yield
    Singleton.purge()


def service() -> InMemoryBackgroundTaskService:
    return InMemoryBackgroundTaskService()


async def test_a_registered_task_runs_and_its_result_is_awaited() -> None:
    tasks = service()

    async def double(value: int) -> int:
        return value * 2

    tasks.register("double", double)
    handle = tasks.enqueue("double", 21)

    assert await tasks.aget_result(handle) == 42


async def test_the_decorator_registers_and_returns_a_callable_reference() -> None:
    tasks = service()

    @tasks.task("greet")
    async def greet(name: str) -> str:
        return f"hello {name}"

    assert greet.name == "greet"
    assert await greet("world") == "hello world"
    assert await tasks.aget_result(await greet.aenqueue("world")) == "hello world"


async def test_a_handle_identifies_the_task_and_its_backend() -> None:
    tasks = service()

    async def noop() -> None:
        return None

    tasks.register("noop", noop)
    handle = tasks.enqueue("noop", metadata={"conversation": "abc"})

    assert handle.name == "noop"
    assert handle.backend is BackgroundTaskBackend.IN_MEMORY
    assert handle.metadata == {"conversation": "abc"}
    await tasks.aget_result(handle)


async def test_each_enqueue_of_the_same_task_gets_its_own_id() -> None:
    tasks = service()

    async def noop() -> None:
        return None

    tasks.register("noop", noop)
    first, second = tasks.enqueue("noop"), tasks.enqueue("noop")

    assert first.id != second.id
    await asyncio.gather(tasks.aget_result(first), tasks.aget_result(second))


def test_registering_a_different_function_under_a_taken_name_is_refused() -> None:
    tasks = service()

    async def first() -> None: ...

    async def second() -> None: ...

    tasks.register("job", first)

    with pytest.raises(ValueError, match="already registered"):
        tasks.register("job", second)


def test_a_replacement_is_allowed_when_asked_for() -> None:
    tasks = service()

    async def first() -> None: ...

    async def second() -> None: ...

    tasks.register("job", first)
    reference = tasks.register("job", second, replace=True)

    assert reference.name == "job"


def test_enqueueing_an_unregistered_name_says_which_one() -> None:
    with pytest.raises(KeyError, match="nothing"):
        service().enqueue("nothing")


def test_a_synchronous_function_cannot_be_registered_as_a_task() -> None:
    tasks = service()
    tasks.register("sync", lambda: 1)

    with pytest.raises(TypeError, match="non-awaitable"):
        tasks.enqueue("sync")


async def test_a_task_that_raises_is_reported_as_failed() -> None:
    tasks = service()

    async def explodes() -> None:
        raise RuntimeError("inside the task")

    tasks.register("explodes", explodes)
    handle = tasks.enqueue("explodes")

    with pytest.raises(RuntimeError, match="inside the task"):
        await tasks.aget_result(handle)


async def test_a_finished_task_reports_success() -> None:
    tasks = service()

    async def noop() -> str:
        return "done"

    tasks.register("noop", noop)
    handle = tasks.enqueue("noop")
    await tasks.aget_result(handle)

    with pytest.raises(KeyError):
        tasks.get_status(handle)


async def test_a_running_task_reports_running_and_refuses_a_sync_result() -> None:
    tasks = service()
    gate = asyncio.Event()

    async def waits() -> str:
        await gate.wait()
        return "done"

    tasks.register("waits", waits)
    handle = tasks.enqueue("waits")
    await asyncio.sleep(0)

    assert tasks.get_status(handle) is TaskStatus.RUNNING
    with pytest.raises(RuntimeError, match="not done"):
        tasks.get_result(handle)

    gate.set()
    await tasks.aget_result(handle)


async def test_a_cancelled_task_is_cancelled_and_then_forgotten() -> None:
    tasks = service()

    async def forever() -> None:
        await asyncio.Event().wait()

    tasks.register("forever", forever)
    handle = tasks.enqueue("forever")
    await asyncio.sleep(0)

    assert tasks.get_status(handle) is TaskStatus.RUNNING
    assert tasks.cancel(handle) is True

    with pytest.raises(asyncio.CancelledError):
        await tasks.aget_result(handle)

    with pytest.raises(KeyError):
        tasks.get_status(handle)


async def test_cancelling_a_finished_task_changes_nothing() -> None:
    tasks = service()

    async def noop() -> None:
        return None

    tasks.register("noop", noop)
    handle = tasks.enqueue("noop")
    await tasks.aget_result(handle)

    assert tasks.cancel(handle) is False


def test_cancelling_an_unknown_id_is_false_rather_than_an_error() -> None:
    assert service().cancel(uuid.uuid4()) is False


async def test_cancelling_by_name_stops_every_live_instance() -> None:
    tasks = service()

    async def forever() -> None:
        await asyncio.Event().wait()

    tasks.register("forever", forever)
    handles = [tasks.enqueue("forever") for _ in range(3)]
    await asyncio.sleep(0)

    assert await tasks.acancel_by_name("forever") == 3
    await asyncio.gather(
        *(tasks.aget_result(h) for h in handles), return_exceptions=True
    )


async def test_a_raw_coroutine_can_be_spawned_without_registration() -> None:
    tasks = service()
    seen: list[str] = []

    async def closure() -> str:
        seen.append("ran")
        return "ok"

    handle = tasks.spawn_coroutine(closure(), name="sse-pump")

    assert handle.name == "sse-pump"
    assert await tasks.aget_result(handle) == "ok"
    assert seen == ["ran"]


async def test_two_tasks_may_not_share_one_id() -> None:
    tasks = service()
    task_id = uuid.uuid4()

    async def forever() -> None:
        await asyncio.Event().wait()

    first = forever()
    second = forever()
    handle = tasks.spawn_coroutine(first, name="forever", task_id=task_id)
    await asyncio.sleep(0)

    try:
        with pytest.raises(ValueError, match="already running"):
            tasks.spawn_coroutine(second, name="forever", task_id=task_id)
    finally:
        second.close()

    tasks.cancel(handle)


async def test_an_explicit_id_is_the_one_the_handle_carries() -> None:
    tasks = service()
    task_id = uuid.uuid4()

    async def noop() -> None:
        return None

    tasks.register("noop", noop)
    handle = tasks.enqueue("noop", task_id=task_id)

    assert handle.id == task_id
    await tasks.aget_result(handle)


async def test_a_result_that_takes_too_long_times_out() -> None:
    tasks = service()

    async def forever() -> None:
        await asyncio.Event().wait()

    tasks.register("forever", forever)
    handle = tasks.enqueue("forever")

    with pytest.raises(TimeoutError):
        await tasks.aget_result(handle, timeout=0.01)

    tasks.cancel(handle)


async def test_shutdown_cancels_everything_still_running() -> None:
    tasks = service()

    async def forever() -> None:
        await asyncio.Event().wait()

    tasks.register("forever", forever)
    handles = [tasks.enqueue("forever") for _ in range(2)]
    await asyncio.sleep(0)

    await tasks.ashutdown(timeout=1.0)

    for handle in handles:
        with pytest.raises(KeyError):
            tasks.get_status(handle)


async def test_shutdown_with_nothing_running_returns_at_once() -> None:
    await service().ashutdown()


async def test_a_status_lookup_for_an_unknown_id_names_it() -> None:
    with pytest.raises(KeyError):
        service().get_status(uuid.uuid4())
