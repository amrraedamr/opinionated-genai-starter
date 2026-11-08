"""The lifespan hook this service contributes to its composition root.

A streaming turn finishes its work in a background task, so a process that
exited without draining them would lose the assistant's half of whatever was
still in flight.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI

from base.services.background_tasks.in_memory import InMemoryBackgroundTaskService
from chat_api.lifespans import background_tasks_lifespan


async def forever() -> None:
    await asyncio.sleep(3600)


def task_for(service: InMemoryBackgroundTaskService, task_id: object) -> asyncio.Task:
    """The live task, captured before shutdown drops it from the registry."""
    return service._tasks[task_id]  # type: ignore[index]


async def test_the_hook_yields_once_and_then_drains() -> None:
    entered = 0
    async with background_tasks_lifespan(FastAPI()):
        entered += 1
    assert entered == 1


async def test_a_task_still_running_at_shutdown_is_stopped() -> None:
    service = InMemoryBackgroundTaskService()
    async with background_tasks_lifespan(FastAPI()):
        handle = service.spawn_coroutine(forever(), name="forever")
        task = task_for(service, handle.id)
        await asyncio.sleep(0)

    assert task.done()
    assert service._tasks == {}


async def test_shutdown_runs_even_when_the_application_failed() -> None:
    """A startup that raised must not leave tasks behind either."""
    service = InMemoryBackgroundTaskService()

    try:
        async with background_tasks_lifespan(FastAPI()):
            handle = service.spawn_coroutine(forever(), name="forever")
            task = task_for(service, handle.id)
            raise RuntimeError("startup failed")
    except RuntimeError:
        pass

    assert task.done()
