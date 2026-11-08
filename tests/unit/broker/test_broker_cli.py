"""`broker-ops`: the dead-letter replay command, driven through its parser.

Someone runs this once, under pressure, with the wrong argument. So what is
pinned is that every argument reaches the broker unchanged, that a malformed
one is refused before anything connects, and that the connection is closed
whether the replay succeeded or not. No subprocess is spawned: `main` reads
`sys.argv`, which a test can set.
"""

import sys
from typing import Any

import pytest

from base.tools import broker_cli


class FakeBroker:
    """Answers the two calls the command makes, and records both."""

    def __init__(self, replayed: int = 0, error: Exception | None = None) -> None:
        self.replayed = replayed
        self.error = error
        self.calls: list[tuple[str, int]] = []
        self.stopped = 0

    async def areplay_dlq(self, topic: str, *, limit: int = 100) -> int:
        self.calls.append((topic, limit))
        if self.error is not None:
            raise self.error
        return self.replayed

    async def astop(self) -> None:
        self.stopped += 1


def run_cli(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], broker: Any = None
) -> None:
    """Run `main` as the console script would, with the broker replaced."""

    def build() -> Any:
        if broker is None:
            raise AssertionError("a broker was built for arguments that are invalid")
        return broker

    monkeypatch.setattr(sys, "argv", ["broker-ops", *argv])
    monkeypatch.setattr(broker_cli, "RedisStreamsBroker", build)
    broker_cli.main()


def test_a_replay_reports_how_many_messages_were_moved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(monkeypatch, ["replay-dlq", "ingestion.failed"], FakeBroker(replayed=3))

    assert capsys.readouterr().out == "replayed=3\n"


def test_an_empty_dead_letter_queue_is_reported_rather_than_staying_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(monkeypatch, ["replay-dlq", "ingestion.failed"], FakeBroker(replayed=0))

    assert capsys.readouterr().out == "replayed=0\n"


def test_the_topic_reaches_the_broker_exactly_as_it_was_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeBroker()

    run_cli(monkeypatch, ["replay-dlq", "تقارير.failed"], broker)

    assert broker.calls == [("تقارير.failed", 100)]


def test_a_limit_that_is_not_given_defaults_to_a_hundred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeBroker()

    run_cli(monkeypatch, ["replay-dlq", "topic"], broker)

    assert broker.calls == [("topic", 100)]


def test_an_explicit_limit_replaces_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeBroker()

    run_cli(monkeypatch, ["replay-dlq", "topic", "--limit", "5"], broker)

    assert broker.calls == [("topic", 5)]


def test_a_limit_of_zero_is_passed_through_rather_than_read_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeBroker()

    run_cli(monkeypatch, ["replay-dlq", "topic", "--limit", "0"], broker)

    assert broker.calls == [("topic", 0)]


def test_a_negative_limit_reaches_the_broker_unchallenged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned as it stands: the parser checks the type, not the range."""
    broker = FakeBroker()

    run_cli(monkeypatch, ["replay-dlq", "topic", "--limit", "-1"], broker)

    assert broker.calls == [("topic", -1)]


def test_the_broker_is_closed_after_a_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = FakeBroker(replayed=1)

    run_cli(monkeypatch, ["replay-dlq", "topic"], broker)

    assert broker.stopped == 1


def test_the_broker_is_closed_even_when_the_replay_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = FakeBroker(error=RuntimeError("stream is gone"))

    with pytest.raises(RuntimeError, match="stream is gone"):
        run_cli(monkeypatch, ["replay-dlq", "topic"], broker)
    assert broker.stopped == 1


def test_a_failed_replay_prints_no_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(RuntimeError):
        run_cli(monkeypatch, ["replay-dlq", "topic"], FakeBroker(error=RuntimeError()))

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["replay-dlq"],
        ["drain-dlq", "topic"],
        ["replay-dlq", "topic", "--limit", "many"],
        ["replay-dlq", "topic", "--limit"],
        ["replay-dlq", "topic", "--unknown"],
    ],
    ids=[
        "nothing",
        "no-topic",
        "unknown-command",
        "limit-not-a-number",
        "bare-limit",
        "unknown-flag",
    ],
)
def test_a_malformed_command_is_refused_before_anything_connects(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, argv)

    assert exit_info.value.code == 2


def test_asking_for_help_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, ["--help"])

    assert exit_info.value.code == 0
    assert "Broker operational commands" in capsys.readouterr().out
