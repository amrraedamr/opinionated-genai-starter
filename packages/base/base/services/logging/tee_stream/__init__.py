import asyncio
import logging
import sys
from collections import deque
from collections.abc import AsyncGenerator
from typing import TextIO

from base.models.definitions.cache import CacheChannels
from base.services.cache.memory import InMemoryCacheService
from base.services.common.meta import KeyedSingleton


class TeeStream:
    def __init__(self, original: TextIO, app_name: str, flush_interval: float = 0.1):
        self._cache_service: InMemoryCacheService = InMemoryCacheService()
        self._channel = CacheChannels.LOGS.for_app(app_name)

        self.original = original
        self._buffer: deque[str] = deque(maxlen=10_000)
        self._flush_task: asyncio.Task | None = None
        self._flush_interval = flush_interval
        self._start_flush_loop()

    def _start_flush_loop(self):
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self):
        """One failed flush must not end flushing for the life of the process."""
        while True:
            await asyncio.sleep(self._flush_interval)
            try:
                await self._flush_buffer()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Log flush failed; continuing")

    async def _flush_buffer(self):
        if not self._buffer:
            return
        batch = []
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

    async def stop(self):
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            await self._flush_buffer()  # final drain

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


class TerminalCaptureService(metaclass=KeyedSingleton):
    def __init__(self, app_name: str):
        self._cache_service: InMemoryCacheService = InMemoryCacheService()
        self._app_name = app_name
        self._channel = CacheChannels.LOGS.for_app(app_name)

        self._original_stdout: TextIO
        self._original_stderr: TextIO
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self):
        if self._active:
            return

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        sys.stdout = TeeStream(self._original_stdout, self._app_name)
        sys.stderr = TeeStream(self._original_stderr, self._app_name)

        # Update ALL registered loggers
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    if handler.stream is self._original_stdout:
                        handler.stream = sys.stdout
                    elif handler.stream is self._original_stderr:
                        handler.stream = sys.stderr

        # Don't forget root logger
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream is self._original_stdout:
                    handler.stream = sys.stdout
                elif handler.stream is self._original_stderr:
                    handler.stream = sys.stderr

        self._active = True

    async def stop(self):
        if not self._active:
            return

        if isinstance(sys.stdout, TeeStream):
            await sys.stdout.stop()
        if isinstance(sys.stderr, TeeStream):
            await sys.stderr.stop()

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
