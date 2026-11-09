"""A capability is the only thing a service opts into, and the only thing probed.

The protocol is structural so a service can declare its own without importing
anything from `base`, and `/readyz` is the single place the fleet learns a
subsystem is not usable. Each capability here is pinned on both halves:
what `astartup` builds, and what `aprobe` reports when it cannot.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI

from base.capabilities import Capability, ProbeStatus
from base.capabilities.broker import BrokerDependency
from base.capabilities.nosql import NoSQLDependency
from base.capabilities.prompts import Prompts
from base.models.view_models.health import HealthReport, ProbeResult
from base.models.view_models.prompt import Prompt, PromptTemplate, SystemMessageTemplate
from base.services.common.meta import Singleton
from base.services.configuration.mixins import PromptSettingsMixin
from base.services.prompt.loader import PromptLoader
from tests.conftest import MakeSettings

APP = FastAPI()


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    Singleton.purge()
    yield
    Singleton.purge()


class Answering:
    """A capability that answers however the test says."""

    def __init__(self, name: str, status: ProbeStatus = ProbeStatus.OK) -> None:
        self.name = name
        self.status = status
        self.events: list[str] = []

    async def astartup(self, app: FastAPI) -> None:
        self.events.append("startup")

    async def ashutdown(self, app: FastAPI) -> None:
        self.events.append("shutdown")

    async def aprobe(self) -> ProbeResult:
        return ProbeResult(name=self.name, status=self.status)


def test_anything_with_the_four_members_is_a_capability() -> None:
    assert isinstance(Answering("x"), Capability)


def test_something_missing_a_probe_is_not_a_capability() -> None:
    class Incomplete:
        name = "incomplete"

        async def astartup(self, app: FastAPI) -> None: ...

        async def ashutdown(self, app: FastAPI) -> None: ...

    assert not isinstance(Incomplete(), Capability)


def test_the_shipped_capabilities_satisfy_the_protocol() -> None:
    from base.capabilities.sql import SQLDependency
    from base.capabilities.terminal_capture import TerminalCapture

    assert isinstance(SQLDependency(), Capability)
    assert isinstance(Prompts(), Capability)
    assert isinstance(TerminalCapture("svc"), Capability)


def test_the_worst_probe_decides_the_report() -> None:
    assert (
        HealthReport.aggregate(
            [
                ProbeResult(name="a", status=ProbeStatus.OK),
                ProbeResult(name="b", status=ProbeStatus.DEGRADED),
                ProbeResult(name="c", status=ProbeStatus.FAILED),
            ]
        ).status
        is ProbeStatus.FAILED
    )


def test_a_degraded_probe_does_not_fail_the_report() -> None:
    assert (
        HealthReport.aggregate(
            [
                ProbeResult(name="a", status=ProbeStatus.OK),
                ProbeResult(name="b", status=ProbeStatus.DEGRADED),
            ]
        ).status
        is ProbeStatus.DEGRADED
    )


def test_a_service_with_no_capabilities_reports_ok() -> None:
    assert HealthReport.aggregate([]).status is ProbeStatus.OK


# Broker


class StubBroker:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy

    async def ahealth_check(self) -> bool:
        return self.healthy


async def test_a_reachable_broker_probes_ok() -> None:
    result = await BrokerDependency(StubBroker(True)).aprobe()  # type: ignore[arg-type]

    assert result.name == "broker"
    assert result.status is ProbeStatus.OK
    assert result.detail is None


async def test_an_unreachable_broker_fails_readiness_with_a_reason() -> None:
    result = await BrokerDependency(StubBroker(False)).aprobe()  # type: ignore[arg-type]

    assert result.status is ProbeStatus.FAILED
    assert result.detail == "broker unreachable"


async def test_the_broker_capability_owns_no_lifecycle_of_its_own() -> None:
    capability = BrokerDependency(StubBroker(True))  # type: ignore[arg-type]
    await capability.astartup(APP)
    await capability.ashutdown(APP)


# NoSQL


class StubNoSQL:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy
        self.disposed = False

    async def ahealth_check(self) -> bool:
        return self.healthy

    async def adispose(self) -> None:
        self.disposed = True


async def test_a_reachable_document_database_probes_ok() -> None:
    result = await NoSQLDependency(StubNoSQL(True)).aprobe()  # type: ignore[arg-type]

    assert (result.name, result.status) == ("nosql", ProbeStatus.OK)


async def test_an_unreachable_document_database_says_so() -> None:
    result = await NoSQLDependency(StubNoSQL(False)).aprobe()  # type: ignore[arg-type]

    assert result.status is ProbeStatus.FAILED
    assert result.detail == "document database unreachable"


async def test_shutting_down_disposes_the_document_database() -> None:
    service = StubNoSQL(True)

    await NoSQLDependency(service).ashutdown(APP)  # type: ignore[arg-type]

    assert service.disposed is True


# Prompts


def register(key: str) -> None:
    PromptLoader().register(
        Prompt(
            key=key,
            name=key,
            description="d",
            template=PromptTemplate.from_messages(
                [SystemMessageTemplate(content="You are terse.")]
            ),
        )
    )


async def test_a_service_with_no_prompts_is_degraded_rather_than_failed() -> None:
    result = await Prompts().aprobe()

    assert result.status is ProbeStatus.DEGRADED
    assert result.detail == "No prompts registered."


async def test_loaded_prompts_are_counted_in_the_probe() -> None:
    register("answer")
    register("summarise")

    result = await Prompts().aprobe()

    assert result.status is ProbeStatus.OK
    assert result.detail == "2 prompts loaded."


async def test_starting_the_prompts_capability_initializes_the_registry(
    make_settings: MakeSettings,
) -> None:
    make_settings(PromptSettingsMixin, load_prompts_from_db=False)
    capability = Prompts()

    await capability.astartup(APP)

    assert PromptLoader()._templates_loaded is True
    await capability.ashutdown(APP)
