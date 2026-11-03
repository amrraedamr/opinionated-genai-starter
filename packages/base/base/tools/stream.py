import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from base.models.view_models.streams import (
    AnyStreamMessage,
    StreamEnd,
    StreamError,
    StreamModel,
)


class StreamIterator:
    def __init__(
        self,
        queue: asyncio.Queue[AnyStreamMessage | str] | None = None,
        source: AsyncIterator[AnyStreamMessage | str] | None = None,
        retry: int = 15000,
    ) -> None:
        self._queue = (
            queue
            if queue
            else (None if source else asyncio.Queue[AnyStreamMessage | str]())
        )
        self._source = source
        self._end = False
        self._retry = retry
        self._retry_sent = False

    def __aiter__(self) -> "StreamIterator":
        return self

    async def __anext__(self) -> dict[str, str | int] | str:
        if self._end:
            raise StopAsyncIteration

        if self._source is not None:
            try:
                message = await self._source.__anext__()
            except StopAsyncIteration:
                self._end = True
                raise
        elif self._queue is not None:
            message = await self._queue.get()
        else:
            raise RuntimeError("StreamIterator must have either a source or a queue")

        if isinstance(message, StreamModel) and not message.id:
            message.id = str(uuid4())

        if isinstance(message, StreamError | StreamEnd):
            self._end = True

        return (
            self._return_response(message)
            if isinstance(message, StreamModel)
            else message
        )

    def _return_response(self, message: StreamModel) -> dict[str, str | int]:
        retry: int | None = None
        if not self._retry_sent:
            retry = self._retry
            self._retry_sent = True

        return self._format_response(
            message.serialize(by_alias=False, exclude_none=True),
            retry=retry,
        )

    @classmethod
    def _format_response(
        cls, data: dict[str, Any], retry: int | None = None
    ) -> dict[str, str | int]:
        response: dict[str, str | int] = {
            "event": "stream",
            "data": json.dumps(data),
        }
        if retry:
            response["retry"] = retry
        return response

    # Async

    async def asend(self, value: AnyStreamMessage) -> None:
        if self._queue is None:
            raise RuntimeError("Cannot send to a source-based StreamIterator")
        await self._queue.put(value)

    async def aend(self, end_stream_model: StreamEnd | None = None) -> None:
        self._end = True
        if end_stream_model:
            await self.asend(end_stream_model)

    async def aerror(self, error_stream_model: StreamError | None = None) -> None:
        self._end = True
        if error_stream_model:
            await self.asend(error_stream_model)

    # Sync

    def send(self, value: AnyStreamMessage) -> None:
        if self._queue is None:
            raise RuntimeError("Cannot send to a source-based StreamIterator")
        self._queue.put_nowait(value)

    def end(self, end_stream_model: StreamEnd | None = None) -> None:
        self._end = True
        if end_stream_model:
            self.send(end_stream_model)

    def error(self, error_stream_model: StreamError | None = None) -> None:
        self._end = True
        if error_stream_model:
            self.send(error_stream_model)
