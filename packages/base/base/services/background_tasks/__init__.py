from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Self
from uuid import UUID

from base.models.definitions.background_tasks import TaskStatus
from base.models.view_models.background_tasks import TaskHandle


class AbstractBackgroundTaskService(ABC):
    """Abstract class for background task service.

    Distributed-safe API: tasks must be registered by name and invoked with
    pickleable args. Each `enqueue` produces a unique `task_id` (TaskHandle.id),
    even when the same `task_name` is reused.

    In-memory backend may expose extra escape hatches for raw coroutines
    (see InMemoryBackgroundTaskService.spawn_coroutine).
    """

    # Registration

    @abstractmethod
    def register(
        self,
        name: str,
        func: Callable[..., Awaitable[Any] | Any],
        *,
        replace: bool = False,
    ) -> "RegisteredTask[Self]":
        """Register a task callable under a name. Returns a typed ref.

        Raises if `name` is already bound to a different function unless
        `replace=True`. Does NOT affect already-running task instances that
        share this name — only updates the registry for future `enqueue` calls.
        """

    def task(
        self,
        name: str,
        *,
        replace: bool = False,
    ) -> Callable[[Callable[..., Awaitable[Any] | Any]], "RegisteredTask[Self]"]:
        """Decorator sugar over `register`.

        Usage:
            @service.task("send_message_stream")
            async def send_message_stream(conv_id: str): ...
        """

        def decorator(
            func: Callable[..., Awaitable[Any] | Any],
        ) -> "RegisteredTask[Self]":
            return self.register(name, func, replace=replace)

        return decorator

    # Enqueue

    @abstractmethod
    def enqueue(
        self,
        task_name: str,
        *args: Any,
        task_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TaskHandle[Any]:
        """Enqueue a registered task. Auto-generates `task_id` when omitted."""

    @abstractmethod
    async def aenqueue(
        self,
        task_name: str,
        *args: Any,
        task_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TaskHandle[Any]:
        """Asynchronously enqueue a registered task."""

    # Cancel

    @abstractmethod
    def cancel(self, handle_or_id: TaskHandle[Any] | UUID) -> bool:
        """Cancel a single task by handle or id. True if cancellation scheduled."""

    @abstractmethod
    async def acancel(self, handle_or_id: TaskHandle[Any] | UUID) -> bool:
        """Asynchronously cancel a single task by handle or id."""

    @abstractmethod
    def cancel_by_name(self, task_name: str) -> int:
        """Cancel all live tasks sharing `task_name`. Returns count cancelled."""

    @abstractmethod
    async def acancel_by_name(self, task_name: str) -> int:
        """Asynchronously cancel all live tasks sharing `task_name`."""

    # Status / Result

    @abstractmethod
    def get_status(self, handle_or_id: TaskHandle[Any] | UUID) -> TaskStatus:
        """Get current status of a task."""

    @abstractmethod
    async def aget_status(self, handle_or_id: TaskHandle[Any] | UUID) -> TaskStatus:
        """Asynchronously get current status of a task."""

    @abstractmethod
    def get_result(self, handle_or_id: TaskHandle[Any] | UUID) -> Any:
        """Return task result if task is already done. Raises if still running.

        Use `aget_result` to await completion. Raises task exception if failed.
        """

    @abstractmethod
    async def aget_result(
        self,
        handle_or_id: TaskHandle[Any] | UUID,
        timeout: float | None = None,
    ) -> Any:
        """Await task result. Raises task exception if task failed."""

    # Lifecycle

    @abstractmethod
    async def ashutdown(self, *, timeout: float | None = 5.0) -> None:
        """Cancel all live tasks and await completion. Use on app shutdown."""

    # Internals

    @staticmethod
    def _resolve_id(handle_or_id: TaskHandle[Any] | UUID) -> UUID:
        return handle_or_id.id if isinstance(handle_or_id, TaskHandle) else handle_or_id


class RegisteredTask[S: AbstractBackgroundTaskService]:
    """Typed reference to a registered background task.

    Generic over the service that produced it so the dev sees, e.g.,
    `RegisteredTask[InMemoryBackgroundTaskService]` and can inspect the
    underlying service for backend-specific methods.

    Returned by `service.register()` or `@service.task(name)`. Acts as a
    callable wrapper around the underlying function (direct call works for
    both sync and async functions, with normal Python call semantics) and
    provides `enqueue`/`aenqueue` and `cancel_all`/`acancel_all` bound to
    the owning service.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., Awaitable[Any] | Any],
        service: S,
    ) -> None:
        self.name: str = name
        self._func: Callable[..., Awaitable[Any] | Any] = func
        self._service: S = service

    @property
    def service(self) -> S:
        return self._service

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the underlying function directly, bypassing the queue.

        Returns the function's natural result: a value for sync functions,
        a coroutine for async functions (caller awaits).
        """
        return self._func(*args, **kwargs)

    def enqueue(
        self,
        *args: Any,
        task_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TaskHandle[Any]:
        return self._service.enqueue(
            self.name, *args, task_id=task_id, metadata=metadata, **kwargs
        )

    async def aenqueue(
        self,
        *args: Any,
        task_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TaskHandle[Any]:
        return await self._service.aenqueue(
            self.name, *args, task_id=task_id, metadata=metadata, **kwargs
        )

    def cancel_all(self) -> int:
        return self._service.cancel_by_name(self.name)

    async def acancel_all(self) -> int:
        return await self._service.acancel_by_name(self.name)
