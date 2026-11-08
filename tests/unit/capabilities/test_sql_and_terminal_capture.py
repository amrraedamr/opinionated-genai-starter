"""The two capabilities that own a process-wide resource rather than a client.

`SQLDependency` holds nothing itself — it reaches for the singleton on each
call, so a composition root can register settings after the capability was
built. `TerminalCapture` is the opposite: starting it rewires `sys.stdout` for
the whole process, so the half that matters is that shutdown undoes it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI

from base.capabilities.sql import SQLDependency
from base.capabilities.terminal_capture import TerminalCapture
from base.models.definitions.health import ProbeStatus
from base.services.common.meta import Singleton
from base.services.logging.tee_stream import TeeStream, TerminalCaptureService
from base.services.sql.service import SQLService

APP = FastAPI()


class StubSQL:
    """Answers the two calls the capability makes, and remembers them."""

    def __init__(self, healthy: bool = True, error: Exception | None = None) -> None:
        self.healthy = healthy
        self.error = error
        self.disposed = False

    async def ahealth_check(self) -> bool:
        if self.error is not None:
            raise self.error
        return self.healthy

    async def adispose(self) -> None:
        self.disposed = True


@pytest.fixture
def sql_service() -> Any:
    """Puts a stand-in where `SQLService()` will find it."""

    def _install(healthy: bool = True, error: Exception | None = None) -> StubSQL:
        stub = StubSQL(healthy, error)
        SQLService.replace(stub)  # type: ignore[arg-type]
        return stub

    return _install


@pytest.fixture(autouse=True)
def _streams_restored() -> Iterator[None]:
    """Capture rewires the process, so a failed test must not leave it rewired."""
    stdout, stderr = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = stdout, stderr


# SQL


def test_the_sql_capability_names_itself_after_the_database() -> None:
    assert SQLDependency().name == "database"


async def test_starting_the_sql_capability_builds_no_connection_of_its_own(
    sql_service: Any,
) -> None:
    """Nothing to open at startup: the first query opens the pool."""
    await SQLDependency().astartup(APP)

    assert SQLService not in Singleton._instances


async def test_shutting_down_disposes_the_database_pool(sql_service: Any) -> None:
    stub = sql_service()

    await SQLDependency().ashutdown(APP)

    assert stub.disposed is True


async def test_a_reachable_database_probes_ok(sql_service: Any) -> None:
    sql_service(healthy=True)

    result = await SQLDependency().aprobe()

    assert (result.name, result.status, result.detail) == (
        "database",
        ProbeStatus.OK,
        None,
    )


async def test_an_unreachable_database_fails_readiness_with_a_reason(
    sql_service: Any,
) -> None:
    sql_service(healthy=False)

    result = await SQLDependency().aprobe()

    assert result.status is ProbeStatus.FAILED
    assert result.detail == "database unreachable"


async def test_a_database_that_raises_lets_the_error_out(sql_service: Any) -> None:
    """`/readyz` turns an exception into a failed probe, so this one need not."""
    sql_service(error=RuntimeError("connection reset"))

    with pytest.raises(RuntimeError):
        await SQLDependency().aprobe()


# Terminal capture


def test_the_capture_capability_names_itself() -> None:
    assert TerminalCapture("svc").name == "terminal-capture"


def test_the_capability_holds_the_service_registered_for_its_application() -> None:
    capability = TerminalCapture("svc")

    assert capability.service is TerminalCaptureService("svc")
    assert capability.service is not TerminalCaptureService("other")


async def test_starting_the_capability_turns_capture_on() -> None:
    capability = TerminalCapture("svc")

    await capability.astartup(APP)

    assert capability.service.is_active is True
    assert isinstance(sys.stdout, TeeStream)
    await capability.ashutdown(APP)


async def test_shutting_down_turns_capture_off_and_hands_the_streams_back() -> None:
    original_stdout, original_stderr = sys.stdout, sys.stderr
    capability = TerminalCapture("svc")
    await capability.astartup(APP)

    await capability.ashutdown(APP)

    assert capability.service.is_active is False
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


async def test_shutting_down_something_that_never_started_is_harmless() -> None:
    await TerminalCapture("svc").ashutdown(APP)

    assert TerminalCapture("svc").service.is_active is False


async def test_capture_that_is_running_probes_ok() -> None:
    capability = TerminalCapture("svc")
    await capability.astartup(APP)

    result = await capability.aprobe()

    assert (result.name, result.status, result.detail) == (
        "terminal-capture",
        ProbeStatus.OK,
        None,
    )
    await capability.ashutdown(APP)


async def test_capture_that_is_not_running_is_degraded_rather_than_failed() -> None:
    """A service nobody is tailing is still a service that can take traffic."""
    result = await TerminalCapture("svc").aprobe()

    assert result.status is ProbeStatus.DEGRADED
    assert result.detail == "Terminal capture is not active."


async def test_starting_twice_leaves_one_capture_running() -> None:
    capability = TerminalCapture("svc")
    await capability.astartup(APP)
    installed = sys.stdout

    await capability.astartup(APP)

    assert sys.stdout is installed
    await capability.ashutdown(APP)
