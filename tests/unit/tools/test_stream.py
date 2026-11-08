"""`StreamIterator`: server-sent events out of a queue or an upstream iterator.

The interesting part is termination. An end or an error frame stops the
iterator, so anything queued behind it is never delivered — which is what keeps
a failed generation from being followed by more tokens.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from base.models.view_models.streams import (
    AnyStreamMessage,
    StreamEnd,
    StreamError,
    StreamMessage,
    StreamProgress,
    StreamStart,
)
from base.tools.stream import StreamIterator


def payload(frame: dict[str, str | int]) -> dict[str, Any]:
    """The decoded `data` field of one server-sent event."""
    assert frame["event"] == "stream"
    return json.loads(str(frame["data"]))


async def drain(iterator: StreamIterator) -> list[Any]:
    return [frame async for frame in iterator]


async def test_queued_messages_come_out_in_order() -> None:
    iterator = StreamIterator()
    iterator.send(StreamStart())
    iterator.send(StreamMessage[str](data="one"))
    iterator.send(StreamMessage[str](data="two"))
    iterator.send(StreamEnd())
    frames = await drain(iterator)
    assert [payload(f)["message_type"] for f in frames] == [
        "start",
        "stream",
        "stream",
        "end",
    ]
    assert [payload(f)["data"] for f in frames[1:3]] == ["one", "two"]


async def test_the_retry_hint_is_sent_once_on_the_first_frame_only() -> None:
    iterator = StreamIterator(retry=4321)
    iterator.send(StreamStart())
    iterator.send(StreamEnd())
    frames = await drain(iterator)
    assert frames[0]["retry"] == 4321
    assert "retry" not in frames[1]


async def test_a_message_with_no_id_is_given_one() -> None:
    iterator = StreamIterator()
    iterator.send(StreamStart())
    iterator.send(StreamEnd())
    frames = await drain(iterator)
    assert payload(frames[0])["id"]
    assert payload(frames[0])["id"] != payload(frames[1])["id"]


async def test_a_message_that_already_has_an_id_keeps_it() -> None:
    iterator = StreamIterator()
    iterator.send(StreamStart(id="given"))
    iterator.send(StreamEnd())
    frames = await drain(iterator)
    assert payload(frames[0])["id"] == "given"


async def test_an_end_frame_is_delivered_and_then_stops_the_stream() -> None:
    iterator = StreamIterator()
    iterator.send(StreamEnd())
    iterator.send(StreamMessage[str](data="never read"))
    frames = await drain(iterator)
    assert len(frames) == 1
    assert payload(frames[0])["message_type"] == "end"


async def test_an_error_frame_is_delivered_and_then_stops_the_stream() -> None:
    iterator = StreamIterator()
    iterator.send(StreamError(error_code=500, text="upstream failed"))
    iterator.send(StreamMessage[str](data="never read"))
    frames = await drain(iterator)
    assert len(frames) == 1
    body = payload(frames[0])
    assert body["message_type"] == "error"
    assert body["error_code"] == 500
    assert body["text"] == "upstream failed"


async def test_an_error_part_way_through_keeps_what_came_before_it() -> None:
    iterator = StreamIterator()
    iterator.send(StreamStart())
    iterator.send(StreamMessage[str](data="partial"))
    iterator.send(StreamError(text="boom"))
    iterator.send(StreamMessage[str](data="never read"))
    frames = await drain(iterator)
    assert [payload(f)["message_type"] for f in frames] == ["start", "stream", "error"]


async def test_fields_left_unset_are_dropped_from_the_wire_payload() -> None:
    iterator = StreamIterator()
    iterator.send(StreamProgress(percentage=0.5))
    iterator.send(StreamEnd())
    body = payload((await drain(iterator))[0])
    assert body["percentage"] == 0.5
    assert "label" not in body
    assert "card_type" not in body


async def test_the_wire_payload_keeps_snake_case_field_names() -> None:
    iterator = StreamIterator()
    iterator.send(StreamError(error_code=1, text="x"))
    body = payload((await drain(iterator))[0])
    assert "error_code" in body
    assert "errorCode" not in body


async def test_a_plain_string_is_passed_through_untouched() -> None:
    iterator = StreamIterator()
    iterator.send("data: raw\n\n")  # type: ignore[arg-type]
    iterator.send(StreamEnd())
    frames = await drain(iterator)
    assert frames[0] == "data: raw\n\n"


async def test_ending_the_stream_stops_it_even_with_nothing_queued() -> None:
    iterator = StreamIterator()
    iterator.end()
    assert await drain(iterator) == []


async def test_a_final_frame_passed_to_aend_is_delivered() -> None:
    """The frame has to be queued before the stream is marked closed.

    `__anext__` checks the flag first, so setting it ahead of the queue made
    the parameter dead: a consumer never saw the frame it was handed.
    """
    iterator = StreamIterator()
    await iterator.aend(StreamEnd(id="last"))
    assert len(await drain(iterator)) == 1


async def test_a_final_frame_passed_to_aerror_is_delivered() -> None:
    """The failing case matters more: this is how a consumer learns why."""
    iterator = StreamIterator()
    await iterator.aerror(StreamError(text="failed"))
    assert len(await drain(iterator)) == 1


async def test_the_sync_terminators_deliver_their_final_frame_too() -> None:
    """The frame closes the stream when it is read, not when it is sent."""
    ended = StreamIterator()
    ended.end(StreamEnd())
    assert len(await drain(ended)) == 1
    assert ended._end is True

    errored = StreamIterator()
    errored.error(StreamError(text="x"))
    assert len(await drain(errored)) == 1
    assert errored._end is True


async def test_a_terminator_with_no_frame_just_closes() -> None:
    iterator = StreamIterator()
    iterator.end()
    assert await drain(iterator) == []
    assert iterator._end is True


async def test_a_terminal_frame_sent_through_send_is_delivered() -> None:
    """The working path: queue the frame, let the iterator notice it is terminal."""
    iterator = StreamIterator()
    iterator.send(StreamEnd(id="last"))
    frames = await drain(iterator)
    assert payload(frames[0])["id"] == "last"


async def test_a_consumer_can_stop_early_and_leave_frames_behind() -> None:
    iterator = StreamIterator()
    iterator.send(StreamStart())
    iterator.send(StreamMessage[str](data="one"))
    iterator.send(StreamEnd())
    first = await iterator.__anext__()
    assert payload(first)["message_type"] == "start"
    assert iterator._queue is not None and iterator._queue.qsize() == 2


async def test_messages_sent_asynchronously_are_delivered_the_same_way() -> None:
    iterator = StreamIterator()
    await iterator.asend(StreamStart())
    await iterator.asend(StreamEnd())
    assert len(await drain(iterator)) == 2


async def test_a_source_backed_iterator_forwards_what_the_source_yields() -> None:
    async def source() -> AsyncIterator[AnyStreamMessage]:
        yield StreamStart()
        yield StreamMessage[str](data="from upstream")

    frames = await drain(StreamIterator(source=source()))
    assert [payload(f)["message_type"] for f in frames] == ["start", "stream"]


async def test_a_source_backed_iterator_refuses_to_be_sent_to() -> None:
    async def source() -> AsyncIterator[AnyStreamMessage]:
        yield StreamStart()

    iterator = StreamIterator(source=source())
    with pytest.raises(RuntimeError, match="source-based"):
        iterator.send(StreamStart())
    with pytest.raises(RuntimeError, match="source-based"):
        await iterator.asend(StreamStart())


async def test_an_error_raised_by_the_source_reaches_the_consumer() -> None:
    async def source() -> AsyncIterator[AnyStreamMessage]:
        yield StreamStart()
        raise RuntimeError("upstream died")

    iterator = StreamIterator(source=source())
    await iterator.__anext__()
    with pytest.raises(RuntimeError, match="upstream died"):
        await iterator.__anext__()


async def test_a_source_that_ends_stops_the_iterator_for_good() -> None:
    async def source() -> AsyncIterator[AnyStreamMessage]:
        yield StreamStart()

    iterator = StreamIterator(source=source())
    await drain(iterator)
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()


async def test_iterating_returns_the_same_object_so_a_for_loop_works() -> None:
    iterator = StreamIterator()
    assert iterator.__aiter__() is iterator


async def test_a_consumer_waits_rather_than_ending_when_the_queue_is_empty() -> None:
    iterator = StreamIterator()

    async def produce() -> None:
        await asyncio.sleep(0.01)
        iterator.send(StreamEnd())

    task = asyncio.create_task(produce())
    frames = await asyncio.wait_for(drain(iterator), timeout=1.0)
    assert len(frames) == 1
    await task
