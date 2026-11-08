"""Terminal capture: stdout and stderr fanned out to whoever is listening.

Two properties carry the whole thing. Output must still reach the terminal —
a tee that swallows what it copies is worse than no tee at all — and the
process must be handed its own streams back, including the streams that
logging handlers are holding, or the second `start` captures nothing.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any

import pytest

from base.models.definitions.cache import CacheChannels
from base.services.cache.memory import InMemoryCacheService
from base.services.logging.tee_stream import TeeStream, TerminalCaptureService

APP = "tee-test"


def channel_for(app_name: str = APP) -> str:
    return CacheChannels.LOGS.for_app(app_name)


async def drain(source: AsyncGenerator[str, None], into: list[str]) -> None:
    async for message in source:
        into.append(message)


async def settle(passes: int = 5) -> None:
    """Let every ready callback run, for an assertion that nothing arrived."""
    for _ in range(passes):
        await asyncio.sleep(0)


async def wait_for(lines: list[str], count: int) -> list[str]:
    """Block until the subscriber has that many lines, rather than guess a sleep."""
    async with asyncio.timeout(5):
        while len(lines) < count:
            await asyncio.sleep(0.005)
    return lines


@pytest.fixture(autouse=True)
def _streams_restored() -> Iterator[None]:
    """A failed test must not leave stdout redirected for the rest of the run."""
    stdout, stderr = sys.stdout, sys.stderr
    named = [logging.getLogger(name) for name in list(logging.root.manager.loggerDict)]
    held = [
        (handler, handler.stream)
        for logger in [*named, logging.root]
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    yield
    for handler, stream in held:
        handler.stream = stream
    sys.stdout, sys.stderr = stdout, stderr


@pytest.fixture
async def tee() -> AsyncIterator[Any]:
    """Tees over a `StringIO`, each one stopped however the test ends."""
    built: list[TeeStream] = []

    def _build(app_name: str = APP, flush_interval: float = 60.0) -> TeeStream:
        stream = TeeStream(io.StringIO(), app_name, flush_interval=flush_interval)
        built.append(stream)
        return stream

    yield _build
    for stream in built:
        await stream.stop()


@pytest.fixture
async def capture(_streams_restored: None) -> AsyncIterator[Any]:
    """Capture services, stopped before the streams are put back."""
    started: list[TerminalCaptureService] = []

    def _service(app_name: str = APP) -> TerminalCaptureService:
        service = TerminalCaptureService(app_name)
        started.append(service)
        return service

    yield _service
    for service in started:
        await service.stop()


@pytest.fixture
async def subscriber() -> AsyncIterator[Any]:
    """Starts a listener and hands back the list it fills."""
    listening: list[asyncio.Task[None]] = []

    async def _subscribe(source: AsyncGenerator[str, None] | str = APP) -> list[str]:
        if isinstance(source, str):
            source = InMemoryCacheService().subscribe(channel_for(source))
        lines: list[str] = []
        listening.append(asyncio.create_task(drain(source, lines)))
        await settle()
        return lines

    yield _subscribe
    for task in listening:
        task.cancel()
    await asyncio.gather(*listening, return_exceptions=True)


@pytest.fixture
def attach() -> Iterator[Any]:
    """Adds a handler for one test and takes it off again."""
    added: list[tuple[logging.Logger, logging.Handler]] = []

    def _attach(logger: logging.Logger, handler: logging.Handler) -> logging.Handler:
        logger.addHandler(handler)
        added.append((logger, handler))
        return handler

    yield _attach
    for logger, handler in added:
        logger.removeHandler(handler)


# The tee itself


async def test_everything_written_still_reaches_the_original_stream(
    tee: Any,
) -> None:
    stream = tee()

    stream.write("visible\n")
    stream.write("   \n")

    assert stream.original.getvalue() == "visible\n   \n"


async def test_a_write_reports_whatever_the_original_stream_reported(
    tee: Any,
) -> None:
    stream = tee()

    assert stream.write("four") == 4


async def test_a_written_line_is_published_once_the_buffer_drains(
    tee: Any, subscriber: Any
) -> None:
    lines = await subscriber()
    stream = tee()

    stream.write("captured\n")
    await stream.stop()

    assert await wait_for(lines, 1) == ["captured\n"]


async def test_a_line_is_published_exactly_as_it_was_written(
    tee: Any, subscriber: Any
) -> None:
    """A partial line has no newline of its own, and is not given one."""
    lines = await subscriber()
    stream = tee()

    stream.write("no newline here")
    await stream.stop()

    assert await wait_for(lines, 1) == ["no newline here"]


@pytest.mark.parametrize("blank", ["\n", "   ", "\t\n", "\r\n", ""])
async def test_a_blank_write_is_passed_through_but_never_published(
    tee: Any, subscriber: Any, blank: str
) -> None:
    lines = await subscriber()
    stream = tee()

    stream.write(blank)
    await stream.stop()
    await settle()

    assert lines == []
    assert stream.original.getvalue() == blank


async def test_stopping_with_an_empty_buffer_publishes_nothing(
    tee: Any, subscriber: Any
) -> None:
    lines = await subscriber()

    await tee().stop()
    await settle()

    assert lines == []


async def test_non_ascii_survives_the_round_trip(tee: Any, subscriber: Any) -> None:
    lines = await subscriber()
    stream = tee()

    stream.write("مرحبا 🐧\n")
    await stream.stop()

    assert await wait_for(lines, 1) == ["مرحبا 🐧\n"]
    assert stream.original.getvalue() == "مرحبا 🐧\n"


async def test_a_very_large_write_is_published_whole(tee: Any, subscriber: Any) -> None:
    lines = await subscriber()
    stream = tee()
    payload = "x" * 1_000_000

    stream.write(payload)
    await stream.stop()

    assert await wait_for(lines, 1) == [payload]


async def test_writes_arrive_in_the_order_they_were_made(
    tee: Any, subscriber: Any
) -> None:
    lines = await subscriber()
    stream = tee()

    for index in range(50):
        stream.write(f"line {index}\n")
    await stream.stop()

    assert await wait_for(lines, 50) == [f"line {index}\n" for index in range(50)]


async def test_concurrent_writers_all_reach_the_channel(
    tee: Any, subscriber: Any
) -> None:
    lines = await subscriber()
    stream = tee()

    async def shout(writer: int) -> None:
        for index in range(20):
            stream.write(f"{writer}:{index}\n")
            await asyncio.sleep(0)

    await asyncio.gather(*(shout(writer) for writer in range(5)))
    await stream.stop()

    assert len(await wait_for(lines, 100)) == 100
    assert set(lines) == {f"{w}:{i}\n" for w in range(5) for i in range(20)}


async def test_the_buffer_holds_only_the_most_recent_ten_thousand_lines(
    tee: Any,
) -> None:
    """An unread channel must not grow until the process runs out of memory."""
    stream = tee()

    for index in range(10_050):
        stream.write(f"line {index}\n")

    assert len(stream._buffer) == 10_000
    assert stream._buffer[0] == "line 50\n"


async def test_the_flush_loop_drains_on_its_own_without_anyone_stopping_it(
    tee: Any, subscriber: Any
) -> None:
    lines = await subscriber()
    stream = tee(flush_interval=0.01)

    stream.write("unattended\n")

    assert await wait_for(lines, 1) == ["unattended\n"]


async def test_flushing_is_handed_to_the_original_stream(tee: Any) -> None:
    flushed: list[bool] = []
    stream = tee()
    stream.original.flush = lambda: flushed.append(True)  # type: ignore[method-assign]

    stream.flush()

    assert flushed == [True]


async def test_an_attribute_the_tee_lacks_comes_from_the_original(tee: Any) -> None:
    stream = tee()
    stream.write("held\n")

    assert stream.getvalue() == "held\n"
    assert stream.writable() is True
    assert stream.isatty() is False


async def test_stopping_a_second_time_is_harmless(tee: Any, subscriber: Any) -> None:
    lines = await subscriber()
    stream = tee()
    stream.write("once\n")

    await stream.stop()
    await stream.stop()

    assert await wait_for(lines, 1) == ["once\n"]


async def test_a_write_after_stopping_still_reaches_the_terminal(tee: Any) -> None:
    """Shutdown ordering is not something the writer can be asked to respect."""
    stream = tee()
    await stream.stop()

    stream.write("during shutdown\n")

    assert stream.original.getvalue() == "during shutdown\n"


async def test_nobody_listening_is_not_an_error(tee: Any) -> None:
    stream = tee()

    stream.write("into the void\n")
    await stream.stop()

    assert stream.original.getvalue() == "into the void\n"


async def test_every_subscriber_gets_its_own_copy(tee: Any, subscriber: Any) -> None:
    listeners = [await subscriber() for _ in range(5)]
    stream = tee()

    stream.write("shared\n")
    await stream.stop()

    for lines in listeners:
        assert await wait_for(lines, 1) == ["shared\n"]


async def test_two_applications_never_read_each_others_output(
    tee: Any, subscriber: Any
) -> None:
    mine = await subscriber("mine")
    theirs = await subscriber("theirs")

    stream = tee("mine")
    stream.write("my line\n")
    await stream.stop()

    assert await wait_for(mine, 1) == ["my line\n"]
    await settle()
    assert theirs == []


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
def test_building_a_tee_outside_a_running_loop_is_refused() -> None:
    """The flush loop is a task, so `start` only works from inside the loop."""
    with pytest.raises(RuntimeError):
        TeeStream(io.StringIO(), APP)


async def test_a_publish_that_fails_is_logged_rather_than_raised(
    tee: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError("cache is gone")

    monkeypatch.setattr(InMemoryCacheService, "apublish_batch", refuse)
    stream = tee()
    stream.write("lost\n")

    await stream.stop()

    assert "Failed to publish logs to cache" in caplog.text


async def test_the_flush_loop_keeps_going_after_one_flush_blew_up(
    tee: Any,
    subscriber: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    publish = InMemoryCacheService.apublish_batch
    remaining = [1]

    async def flaky(
        self: InMemoryCacheService,
        channel: str,
        messages: list[str],
        ttl_seconds: int = 86400,
    ) -> None:
        if remaining[0]:
            remaining[0] -= 1
            raise ValueError("something the flush did not expect")
        await publish(self, channel, messages, ttl_seconds)

    monkeypatch.setattr(InMemoryCacheService, "apublish_batch", flaky)
    lines = await subscriber()
    stream = tee(flush_interval=0.01)

    stream.write("first\n")
    async with asyncio.timeout(5):
        while remaining[0]:
            await asyncio.sleep(0.005)
    stream.write("second\n")

    assert await wait_for(lines, 1) == ["second\n"]
    assert "Log flush failed; continuing" in caplog.text


async def test_being_cancelled_mid_flush_is_not_logged_as_a_failed_flush(
    tee: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Shutdown cancels the loop, and a shutdown is not an incident."""
    publishing = asyncio.Event()

    async def never_finishes(*args: Any, **kwargs: Any) -> None:
        publishing.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(InMemoryCacheService, "apublish_batch", never_finishes)
    stream = tee(flush_interval=0.01)
    stream.write("in flight\n")
    async with asyncio.timeout(5):
        await publishing.wait()

    await stream.stop()

    assert "Log flush failed" not in caplog.text
    assert stream._flush_task.cancelled()


# The capture service


def test_one_capture_service_per_application_name() -> None:
    assert TerminalCaptureService(APP) is TerminalCaptureService(APP)


def test_two_applications_get_a_capture_service_each() -> None:
    assert TerminalCaptureService("one") is not TerminalCaptureService("two")


def test_capture_is_off_until_it_is_started() -> None:
    assert TerminalCaptureService(APP).is_active is False


async def test_starting_puts_a_tee_in_front_of_both_streams(capture: Any) -> None:
    original_stdout, original_stderr = sys.stdout, sys.stderr
    service = capture()

    service.start()

    assert isinstance(sys.stdout, TeeStream)
    assert isinstance(sys.stderr, TeeStream)
    assert sys.stdout.original is original_stdout
    assert sys.stderr.original is original_stderr
    assert service.is_active is True


async def test_stopping_puts_the_original_streams_back(capture: Any) -> None:
    original_stdout, original_stderr = sys.stdout, sys.stderr
    service = capture()
    service.start()

    await service.stop()

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert service.is_active is False


async def test_starting_again_while_active_does_not_stack_a_second_tee(
    capture: Any,
) -> None:
    service = capture()
    service.start()
    installed = sys.stdout

    service.start()

    assert sys.stdout is installed
    assert sys.stdout.original is not installed


async def test_stopping_something_that_never_started_is_harmless(
    capture: Any,
) -> None:
    original_stdout = sys.stdout

    await capture().stop()

    assert sys.stdout is original_stdout


async def test_a_full_cycle_can_be_run_more_than_once(capture: Any) -> None:
    original_stdout = sys.stdout
    service = capture()

    for _ in range(3):
        service.start()
        assert isinstance(sys.stdout, TeeStream)
        await service.stop()
        assert sys.stdout is original_stdout


async def test_what_the_service_prints_reaches_a_subscriber(
    capture: Any, subscriber: Any
) -> None:
    lines = await subscriber()
    service = capture()

    service.start()
    print("printed by the service")
    await service.stop()

    assert await wait_for(lines, 1) == ["printed by the service"]


async def test_what_the_service_writes_to_stderr_shares_the_same_channel(
    capture: Any, subscriber: Any
) -> None:
    lines = await subscriber()
    service = capture()

    service.start()
    print("a warning", file=sys.stderr)
    await service.stop()

    assert await wait_for(lines, 1) == ["a warning"]


async def test_the_channel_belongs_to_one_application_alone(
    capture: Any, subscriber: Any
) -> None:
    theirs = await subscriber("somebody-else")
    mine = await subscriber("ours")
    service = capture("ours")

    service.start()
    print("ours alone")
    await service.stop()

    assert await wait_for(mine, 1) == ["ours alone"]
    await settle()
    assert theirs == []


async def test_a_subscriber_that_joins_late_is_given_the_backlog(
    capture: Any, subscriber: Any
) -> None:
    service = capture()
    service.start()
    print("before anyone was listening")
    await service.stop()

    lines = await subscriber(service.subscribe())
    await settle()

    assert any("before anyone was listening" in line for line in lines)


async def test_a_subscriber_that_goes_away_mid_stream_does_not_stop_the_rest(
    capture: Any, subscriber: Any
) -> None:
    staying = await subscriber()
    leaving = InMemoryCacheService().subscribe(channel_for())
    pulling = asyncio.create_task(drain(leaving, []))
    await settle()
    pulling.cancel()
    await asyncio.gather(pulling, return_exceptions=True)

    service = capture()
    service.start()
    print("still delivered")
    await service.stop()

    assert await wait_for(staying, 1) == ["still delivered"]


async def test_the_service_hands_out_a_tail_of_its_own_channel(
    capture: Any, subscriber: Any
) -> None:
    service = capture()
    lines = await subscriber(service.subscribe())

    service.start()
    print("tailed")
    await service.stop()

    assert await wait_for(lines, 1) == ["tailed"]


async def test_every_tail_gets_its_own_copy(capture: Any, subscriber: Any) -> None:
    service = capture()
    tails = [await subscriber(service.subscribe()) for _ in range(3)]

    service.start()
    print("to everyone")
    await service.stop()

    for lines in tails:
        assert await wait_for(lines, 1) == ["to everyone"]


async def test_a_tail_reads_only_the_channel_of_its_own_application(
    capture: Any, subscriber: Any
) -> None:
    theirs = await subscriber(capture("not-ours").subscribe())
    service = capture("ours")
    mine = await subscriber(service.subscribe())

    service.start()
    print("ours alone")
    await service.stop()

    assert await wait_for(mine, 1) == ["ours alone"]
    await settle()
    assert theirs == []


# Logging handlers


async def test_a_handler_pointed_at_stdout_is_moved_onto_the_tee(
    capture: Any, attach: Any
) -> None:
    handler = attach(logging.root, logging.StreamHandler(sys.stdout))

    capture().start()

    assert handler.stream is sys.stdout
    assert isinstance(handler.stream, TeeStream)


async def test_a_handler_on_a_named_logger_is_moved_too(
    capture: Any, attach: Any
) -> None:
    logger = logging.getLogger("tee.named")
    handler = attach(logger, logging.StreamHandler(sys.stderr))

    capture().start()

    assert handler.stream is sys.stderr


async def test_a_handler_is_moved_back_when_capture_stops(
    capture: Any, attach: Any
) -> None:
    """Left on a stopped tee, a handler writes into a flush loop that is gone."""
    original_stdout = sys.stdout
    handler = attach(logging.root, logging.StreamHandler(original_stdout))
    service = capture()
    service.start()

    await service.stop()

    assert handler.stream is original_stdout


async def test_logging_still_reaches_the_channel_after_capture_is_restarted(
    capture: Any, subscriber: Any, attach: Any
) -> None:
    lines = await subscriber()
    handler = attach(logging.root, logging.StreamHandler(sys.stdout))
    handler.setFormatter(logging.Formatter("%(message)s"))
    service = capture()

    service.start()
    await service.stop()
    service.start()
    logging.getLogger("tee.restart").error("after the restart")
    await service.stop()

    assert "after the restart\n" in await wait_for(lines, 1)


async def test_a_handler_on_an_unrelated_stream_is_left_alone(
    capture: Any, attach: Any
) -> None:
    elsewhere = io.StringIO()
    handler = attach(logging.root, logging.StreamHandler(elsewhere))
    service = capture()

    service.start()
    assert handler.stream is elsewhere
    await service.stop()

    assert handler.stream is elsewhere


async def test_a_handler_that_is_not_a_stream_handler_is_left_alone(
    capture: Any, attach: Any
) -> None:
    received: list[str] = []

    class Collecting(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            received.append(record.getMessage())

    logger = logging.getLogger("tee.collecting")
    attach(logger, Collecting())
    service = capture()

    service.start()
    logger.error("still handled")
    await service.stop()

    assert received == ["still handled"]
