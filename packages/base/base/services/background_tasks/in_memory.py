import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime, timezone
from typing import Any, ClassVar, Self
from uuid import UUID, uuid4

from base.models.definitions.background_tasks import (
    BackgroundTaskBackend,
    TaskStatus,
)
from base.models.view_models.background_tasks import TaskHandle
from base.services.background_tasks import (
    AbstractBackgroundTaskService,
    RegisteredTask,
)
from base.services.common.meta import SingletonABC
from base.services.logging.console import ConsoleLoggerService


class InMemoryBackgroundTaskService(
    AbstractBackgroundTaskService, metaclass=SingletonABC
):
    """In-process asyncio-based background task service.

    Tasks run in the current event loop. State is lost on process restart.
    Suitable for request-scoped fire-and-forget work (e.g. SSE pumps).

    Provides an extra `spawn_coroutine` escape hatch that accepts raw
    coroutines (e.g. closures over request-scope vars) without registration
    — only meaningful in-process and not portable to distributed backends.
    """

    SERVICE_NAME: ClassVar[str] = "In-Memory Background Tasks"
    BACKEND: ClassVar[BackgroundTaskBackend] = BackgroundTaskBackend.IN_MEMORY

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., Awaitable[Any] | Any]] = {}
        self._tasks: dict[UUID, asyncio.Task[Any]] = {}
        self._handles: dict[UUID, TaskHandle[Any]] = {}
        self._logger = ConsoleLoggerService()

    # Registration

    def register(
        self,
        name: str,
        func: Callable[..., Awaitable[Any] | Any],
        *,
        replace: bool = False,
    ) -> RegisteredTask[Self]:
        existing = self._registry.get(name)
        if existing is not None and existing is not func and not replace:
            raise ValueError(
                f"Background task {name!r} is already registered; "
                f"pass replace=True to override"
            )
        self._registry[name] = func
        return RegisteredTask(name=name, func=func, service=self)

    # Enqueue

    def enqueue(
        self,
        task_name: str,
        *args: Any,
        task_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TaskHandle[Any]:
        func = self._registry.get(task_name)
        if func is None:
            raise KeyError(f"No background task registered under name {task_name!r}")
        coro = self._as_coroutine(func, args, kwargs, task_name)
        return self._schedule(coro, name=task_name, task_id=task_id, metadata=metadata)

    async def aenqueue(
        self,
        task_name: str,
        *args: Any,
        task_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TaskHandle[Any]:
        return self.enqueue(
            task_name, *args, task_id=task_id, metadata=metadata, **kwargs
        )

    def spawn_coroutine(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        task_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskHandle[Any]:
        """In-memory only escape hatch for raw coroutines.

        Use when a task body cannot be a registered named function — typically
        a closure over request-scope vars (e.g. an SSE pump bound to a
        per-request StreamIterator). Not portable to distributed backends.
        """
        return self._schedule(coro, name=name, task_id=task_id, metadata=metadata)

    # Cancel

    def cancel(self, handle_or_id: TaskHandle[Any] | UUID) -> bool:
        task_id = self._resolve_id(handle_or_id)
        task = self._tasks.get(task_id)
        if task is None or task.done():
            return False
        return task.cancel()

    async def acancel(self, handle_or_id: TaskHandle[Any] | UUID) -> bool:
        return self.cancel(handle_or_id)

    def cancel_by_name(self, task_name: str) -> int:
        cancelled = 0
        for task_id, handle in list(self._handles.items()):
            if handle.name != task_name:
                continue
            task = self._tasks.get(task_id)
            if task is None or task.done():
                continue
            if task.cancel():
                cancelled += 1
        return cancelled

    async def acancel_by_name(self, task_name: str) -> int:
        return self.cancel_by_name(task_name)

    # Status / Result

    def get_status(self, handle_or_id: TaskHandle[Any] | UUID) -> TaskStatus:
        task_id = self._resolve_id(handle_or_id)
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task id {task_id!s}")
        if task.cancelled():
            return TaskStatus.CANCELLED
        if not task.done():
            coro = task.get_coro()
            cr_frame = getattr(coro, "cr_frame", None)
            if cr_frame is None:
                return TaskStatus.RUNNING
            return TaskStatus.PENDING if cr_frame.f_lasti < 0 else TaskStatus.RUNNING
        return TaskStatus.FAILED if task.exception() is not None else TaskStatus.SUCCESS

    async def aget_status(self, handle_or_id: TaskHandle[Any] | UUID) -> TaskStatus:
        return self.get_status(handle_or_id)

    def get_result(self, handle_or_id: TaskHandle[Any] | UUID) -> Any:
        task_id = self._resolve_id(handle_or_id)
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task id {task_id!s}")
        if not task.done():
            raise RuntimeError(
                f"Task {task_id!s} is not done; use `aget_result` to await"
            )
        return task.result()

    async def aget_result(
        self,
        handle_or_id: TaskHandle[Any] | UUID,
        timeout: float | None = None,
    ) -> Any:
        task_id = self._resolve_id(handle_or_id)
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task id {task_id!s}")
        if timeout is None:
            return await task
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    # Lifecycle

    async def ashutdown(self, *, timeout: float | None = 5.0) -> None:
        live = [t for t in self._tasks.values() if not t.done()]
        if not live:
            return
        for task in live:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*live, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            still_running = sum(1 for t in live if not t.done())
            self._logger.error(
                f"Background task shutdown timed out after {timeout}s; "
                f"{still_running} task(s) still running"
            )

    # Internals

    def _schedule(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None,
        task_id: UUID | None,
        metadata: dict[str, Any] | None,
    ) -> TaskHandle[Any]:
        resolved_id = task_id or uuid4()
        existing = self._tasks.get(resolved_id)
        if existing is not None and not existing.done():
            raise ValueError(f"Background task with id {resolved_id!s} already running")
        handle: TaskHandle[Any] = TaskHandle(
            id=resolved_id,
            name=name,
            backend=self.BACKEND,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        task = asyncio.create_task(coro, name=str(resolved_id))
        self._tasks[resolved_id] = task
        self._handles[resolved_id] = handle
        task.add_done_callback(self._on_task_done)
        return handle

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        task_id = UUID(task.get_name())
        self._tasks.pop(task_id, None)
        handle = self._handles.pop(task_id, None)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            display_name = handle.name if handle and handle.name else str(task_id)
            self._logger.error(
                f"Background task {display_name!r} (id={task_id!s}) raised: "
                f"{exception!r}"
            )

    @staticmethod
    def _as_coroutine(
        func: Callable[..., Awaitable[Any] | Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        task_name: str,
    ) -> Coroutine[Any, Any, Any]:
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return result
        if hasattr(result, "__await__"):

            async def _wrap() -> Any:
                return await result

            return _wrap()
        raise TypeError(
            f"Registered task {task_name!r} returned non-awaitable "
            f"{type(result).__name__}; register an async function"
        )
