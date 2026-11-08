"""A log line is built in two halves: deferred formatting, and a fluent builder.

`LazyLogMessage` exists so a debug line costs nothing when the level is off,
which is only true if the arguments stay unrendered until something asks for
the string. The builder is the other half: one `LogMessage` assembled by name
rather than by argument position.
"""

from __future__ import annotations

import datetime
from enum import Enum

import pytest

from base.services.logging.helper.log_builder import (
    LazyLogMessage,
    LogMessage,
    LogMessageBuilder,
)


class Tag(Enum):
    UPSTREAM = "Upstream"
    RETRY = "Retry"


class Expensive:
    """Something nobody wants rendered unless the line is actually emitted."""

    def __init__(self) -> None:
        self.renders = 0

    def __str__(self) -> str:
        self.renders += 1
        return "expensive"


# Deferred formatting


def test_an_argument_is_not_rendered_until_the_message_is() -> None:
    argument = Expensive()

    message = LazyLogMessage("got %s", argument)

    assert argument.renders == 0
    assert str(message) == "got expensive"
    assert argument.renders == 1


def test_a_message_with_no_arguments_is_its_own_format_string() -> None:
    assert str(LazyLogMessage("nothing to fill in")) == "nothing to fill in"


def test_a_percent_sign_is_left_alone_when_there_is_nothing_to_fill_in() -> None:
    """A bare `%` would raise if the format were applied regardless."""
    assert str(LazyLogMessage("100% done")) == "100% done"


def test_several_arguments_are_filled_in_positionally() -> None:
    assert str(LazyLogMessage("%s took %dms", "retrieval", 42)) == "retrieval took 42ms"


def test_rendering_twice_gives_the_same_line_twice() -> None:
    message = LazyLogMessage("%s", "stable")

    assert str(message) == str(message) == "stable"


def test_a_lazy_message_carries_no_dictionary_of_its_own() -> None:
    """`__slots__` is the point: one of these is created per suppressed line."""
    message = LazyLogMessage("x")

    assert not hasattr(message, "__dict__")
    with pytest.raises(AttributeError):
        message.extra = 1  # type: ignore[attr-defined]


def test_a_format_that_does_not_match_its_arguments_fails_at_rendering() -> None:
    message = LazyLogMessage("%d", "not a number")

    with pytest.raises(TypeError):
        str(message)


# The message


def test_a_fresh_message_starts_empty() -> None:
    message = LogMessage()

    assert message.tags == []
    assert message.message == ""
    assert message.status == ""
    assert message.duration == 0.0


def test_a_message_becomes_a_dictionary_of_every_field() -> None:
    message = LogMessage()
    message.message = "retrieved 12 chunks"
    message.status = 200
    message.duration = 1.5

    payload = message.to_dict()

    assert payload["status"] == 200
    assert payload["message"] == "retrieved 12 chunks"
    assert payload["duration"] == 1.5
    assert payload["tags"] == []
    assert payload["timestamp"]


def test_an_enum_tag_is_stored_as_its_value() -> None:
    message = LogMessage()
    message.tags = [Tag.UPSTREAM, Tag.RETRY]

    assert message.to_dict()["tags"] == ["Upstream", "Retry"]


def test_a_plain_tag_is_kept_exactly_as_it_was_given() -> None:
    message = LogMessage()
    message.tags = ["Upstream", 7]

    assert message.to_dict()["tags"] == ["Upstream", 7]


def test_the_timestamp_is_utc_to_the_second() -> None:
    before = datetime.datetime.now(datetime.UTC).replace(microsecond=0)

    stamped = datetime.datetime.strptime(
        LogMessage().to_dict()["timestamp"], "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=datetime.UTC)

    assert before <= stamped <= before + datetime.timedelta(seconds=5)


def test_the_rendered_line_carries_every_field_in_order() -> None:
    message = (
        LogMessageBuilder()
        .message("it broke")
        .status(503)
        .tags([Tag.UPSTREAM])
        .duration(0.25)
        .build()
    )

    stamp, _, rest = message.to_message().partition(" - ")

    assert datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    assert rest == "503 - it broke - ['Upstream'] - 0.25"


def test_two_messages_do_not_share_a_tag_list() -> None:
    first, second = LogMessage(), LogMessage()

    first.tags.append("mine")

    assert second.tags == []


# The builder


def test_every_step_returns_the_builder_so_calls_can_be_chained() -> None:
    builder = LogMessageBuilder()

    assert builder.message("m") is builder
    assert builder.status(200) is builder
    assert builder.tags(["t"]) is builder
    assert builder.duration(1.0) is builder


def test_the_builder_assembles_everything_it_was_given() -> None:
    message = (
        LogMessageBuilder()
        .message("ingested")
        .status("accepted")
        .tags(["Rag"])
        .duration(12.5)
        .build()
    )

    assert (message.message, message.status, message.duration) == (
        "ingested",
        "accepted",
        12.5,
    )
    assert message.tags == ["Rag"]


def test_tags_accumulate_rather_than_replace() -> None:
    message = LogMessageBuilder().tags(["one"]).tags(["two", "three"]).build()

    assert message.tags == ["one", "two", "three"]


def test_building_twice_hands_back_the_same_message() -> None:
    builder = LogMessageBuilder()

    assert builder.build() is builder.build()


def test_two_builders_do_not_share_a_message() -> None:
    first = LogMessageBuilder().message("first")

    assert LogMessageBuilder().build().message == ""
    assert first.build().message == "first"


def test_a_message_that_was_never_given_anything_still_renders() -> None:
    line = LogMessageBuilder().build().to_message()

    assert line.endswith(" -  -  - [] - 0.0")
