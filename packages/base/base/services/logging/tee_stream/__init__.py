import asyncio
import logging
import sys
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any, TextIO

from base.models.definitions.cache import CacheChannels
from base.services.cache.memory import InMemoryCacheService
from base.services.common.meta import KeyedSingleton


class TeeStream:
    def __init__(
        self, original: TextIO, app_name: str, flush_interval: float = 0.1
    ) -> None:
        self._cache_service: InMemoryCacheService = InMemoryCacheService()
        self._channel = CacheChannels.LOGS.for_app(app_name)

        self.original = original
        self._buffer: deque[str] = deque(maxlen=10_000)
        self._flush_task: asyncio.Task | None = None
        self._flush_interval = flush_interval
        self._start_flush_loop()

    def _start_flush_loop(self) -> None:
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        """One failed flush must not end flushing for the life of the process."""
        while True:
            await asyncio.sleep(self._flush_interval)
            try:
                await self._flush_buffer()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Log flush failed; continuing")

    async def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        batch: list[str] = []
        while self._buffer:
            batch.append(self._buffer.popleft())
        try:
            await self._cache_service.apublish_batch(self._channel, batch)
        except (OSError, RuntimeError) as e:
            logging.exception("Failed to publish logs to cache: %s", e)

    def write(self, text: str) -> int:
        if text and text.strip():
            self._buffer.append(text)
        return self.original.write(text)

    async def stop(self) -> None:
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            await self._flush_buffer()  # final drain

    def flush(self) -> None:
        self.original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


class TerminalCaptureService(metaclass=KeyedSingleton):
    def __init__(self, app_name: str) -> None:
        self._cache_service: InMemoryCacheService = InMemoryCacheService()
        self._app_name = app_name
        self._channel = CacheChannels.LOGS.for_app(app_name)

        self._original_stdout: TextIO
        self._original_stderr: TextIO
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def _retarget_handlers(
        self,
        from_stdout: TextIO,
        from_stderr: TextIO,
        to_stdout: TextIO,
        to_stderr: TextIO,
    ) -> None:
        """Move every stream handler holding one of these streams onto its replacement.

        A handler keeps the stream it was built with, so one that is not moved
        back on the way out goes on writing into a tee whose flush loop has
        already been cancelled — and its output never reaches the channel again.
        """
        named = [
            logging.getLogger(name) for name in list(logging.root.manager.loggerDict)
        ]
        for logger in [*named, logging.root]:
            for handler in logger.handlers:
                if not isinstance(handler, logging.StreamHandler):
                    continue
                if handler.stream is from_stdout:
                    handler.stream = to_stdout
                elif handler.stream is from_stderr:
                    handler.stream = to_stderr

    def start(self) -> None:
        if self._active:
            return

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        sys.stdout = TeeStream(self._original_stdout, self._app_name)
        sys.stderr = TeeStream(self._original_stderr, self._app_name)

        self._retarget_handlers(
            self._original_stdout, self._original_stderr, sys.stdout, sys.stderr
        )
        self._active = True

    async def stop(self) -> None:
        if not self._active:
            return

        captured_stdout, captured_stderr = sys.stdout, sys.stderr
        if isinstance(captured_stdout, TeeStream):
            await captured_stdout.stop()
        if isinstance(captured_stderr, TeeStream):
            await captured_stderr.stop()

        self._retarget_handlers(
            captured_stdout,
            captured_stderr,
            self._original_stdout,
            self._original_stderr,
        )
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._active = False

    async def subscribe(self) -> AsyncGenerator[str, None]:
        pubsub_gen = self._cache_service.subscribe(self._channel)
        history = await self._cache_service.aget_list_raw(f"{self._channel}:history")
        for message in history:
            yield message
        async for message in pubsub_gen:
            yield message
