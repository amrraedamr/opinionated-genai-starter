"""The two handlers this worker is composed of.

They are deliberately tolerant: a payload missing the field they log is still
a delivery the broker has to be told about, and a handler that raised on it
would be redelivered until it was dead-lettered. What is pinned here is that
they run, that they count, and that what they hand the logger is a template
its arguments actually fill.
"""

from __future__ import annotations

import inspect
import uuid
from typing import Any

import pytest

from contracts.events import Event
from contracts.v1 import chat, identity
from ingestion_worker import handlers
from tests.unit.ingestion_worker.conftest import (
    PRODUCER,
    conversation_created,
    user_registered,
)


class RecordingLogger:
    """Keeps the template and arguments a handler passed, instead of printing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def info(self, message: str, *args: Any) -> None:
        self.calls.append((message, args))

    @property
    def last(self) -> tuple[str, tuple[Any, ...]]:
        assert self.calls, "nothing was logged"
        return self.calls[-1]


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> RecordingLogger:
    recorder = RecordingLogger()
    monkeypatch.setattr(handlers, "logger", recorder)
    return recorder


HANDLERS: list[Any] = [
    handlers.ahandle_user_registered,
    handlers.ahandle_conversation_created,
]


# Shape


@pytest.mark.parametrize("handler", HANDLERS)
def test_every_handler_is_a_coroutine_function(handler: Any) -> None:
    assert inspect.iscoroutinefunction(handler)


@pytest.mark.parametrize("handler", HANDLERS)
async def test_a_handler_answers_with_nothing(
    handler: Any, logged: RecordingLogger
) -> None:
    """The inbox reads the outcome from whether it raised, not from a value."""
    assert await handler(user_registered()) is None


# What they count


async def test_a_handled_user_registration_is_counted(
    logged: RecordingLogger,
) -> None:
    await handlers.ahandle_user_registered(user_registered())
    assert handlers.processed_count() == 1


async def test_a_handled_conversation_is_counted(logged: RecordingLogger) -> None:
    await handlers.ahandle_conversation_created(conversation_created())
    assert handlers.processed_count() == 1


async def test_the_count_is_the_services_own_rather_than_one_per_topic(
    logged: RecordingLogger,
) -> None:
    await handlers.ahandle_user_registered(user_registered())
    await handlers.ahandle_conversation_created(conversation_created())
    assert handlers.processed_count() == 2


async def test_the_count_rises_once_per_delivery(logged: RecordingLogger) -> None:
    for _ in range(5):
        await handlers.ahandle_user_registered(user_registered())
    assert handlers.processed_count() == 5


def test_nothing_handled_counts_as_nothing_processed() -> None:
    assert handlers.processed_count() == 0


# What they log


async def test_a_user_registration_logs_the_user_it_was_told_about(
    logged: RecordingLogger,
) -> None:
    delivered = user_registered(user_id="33333333-3333-3333-3333-333333333333")
    await handlers.ahandle_user_registered(delivered)
    message, args = logged.last
    assert args[0] == "33333333-3333-3333-3333-333333333333"
    assert "user registered" in message


async def test_a_conversation_logs_the_conversation_it_was_told_about(
    logged: RecordingLogger,
) -> None:
    delivered = conversation_created(
        conversation_id="44444444-4444-4444-4444-444444444444"
    )
    await handlers.ahandle_conversation_created(delivered)
    message, args = logged.last
    assert args[0] == "44444444-4444-4444-4444-444444444444"
    assert "conversation created" in message


@pytest.mark.parametrize(
    ("handler", "build"),
    [
        (handlers.ahandle_user_registered, user_registered),
        (handlers.ahandle_conversation_created, conversation_created),
    ],
)
async def test_a_handler_logs_the_correlation_id_it_was_given(
    handler: Any, build: Any, logged: RecordingLogger
) -> None:
    """A correlation id is what ties this log line back to the request."""
    correlation = uuid.uuid4()
    delivered = build()
    delivered.meta.correlation_id = correlation
    await handler(delivered)
    assert logged.last[1][1] == correlation


@pytest.mark.parametrize(
    ("handler", "build"),
    [
        (handlers.ahandle_user_registered, user_registered),
        (handlers.ahandle_conversation_created, conversation_created),
    ],
)
async def test_an_uncorrelated_event_logs_no_correlation_id(
    handler: Any, build: Any, logged: RecordingLogger
) -> None:
    await handler(build())
    assert logged.last[1][1] is None


@pytest.mark.parametrize(
    ("handler", "build"),
    [
        (handlers.ahandle_user_registered, user_registered),
        (handlers.ahandle_conversation_created, conversation_created),
    ],
)
async def test_the_template_a_handler_logs_is_filled_by_its_own_arguments(
    handler: Any, build: Any, logged: RecordingLogger
) -> None:
    """Logging swallows a placeholder mismatch, so it is checked here instead."""
    await handler(build())
    message, args = logged.last
    assert message % args


# Payloads that do not say what the topic promised


@pytest.mark.parametrize("handler", HANDLERS)
async def test_an_empty_payload_is_still_handled(
    handler: Any, logged: RecordingLogger
) -> None:
    await handler(Event.of("any.topic", {}, producer=PRODUCER))
    assert handlers.processed_count() == 1


async def test_a_registration_payload_without_a_user_logs_nothing_for_it(
    logged: RecordingLogger,
) -> None:
    await handlers.ahandle_user_registered(
        Event.of(identity.USER_REGISTERED, {"email": "a@b.c"}, producer=PRODUCER)
    )
    assert logged.last[1][0] is None


async def test_a_conversation_payload_carrying_the_wrong_event_is_still_handled(
    logged: RecordingLogger,
) -> None:
    """A payload that does not match its topic must not become a poison pill."""
    mismatched = Event.of(
        chat.CONVERSATION_CREATED,
        {"user_id": "11111111-1111-1111-1111-111111111111"},
        producer=PRODUCER,
    )
    await handlers.ahandle_conversation_created(mismatched)
    assert logged.last[1][0] is None
    assert handlers.processed_count() == 1


@pytest.mark.parametrize("handler", HANDLERS)
async def test_a_payload_the_handler_does_not_read_is_ignored_rather_than_refused(
    handler: Any, logged: RecordingLogger
) -> None:
    """Producers add fields; a consumer that refused them would stop at the first."""
    delivered = Event.of(
        "any.topic", {"something": "new", "nested": {"a": [1, 2]}}, producer=PRODUCER
    )
    await handler(delivered)
    assert handlers.processed_count() == 1


@pytest.mark.parametrize("handler", HANDLERS)
async def test_a_handler_leaves_the_event_it_was_given_alone(
    handler: Any, logged: RecordingLogger
) -> None:
    delivered = user_registered()
    before = delivered.model_dump(mode="json")
    await handler(delivered)
    assert delivered.model_dump(mode="json") == before
