from fastapi import FastAPI

from base.models.definitions.health import ProbeStatus
from base.models.view_models.health import ProbeResult
from base.services.broker import AbstractBroker


class BrokerDependency:
    """Readiness for the event transport this service publishes or consumes on."""

    name = "broker"

    def __init__(self, broker: AbstractBroker) -> None:
        self._broker = broker

    async def astartup(self, app: FastAPI) -> None:
        del app

    async def ashutdown(self, app: FastAPI) -> None:
        del app

    async def aprobe(self) -> ProbeResult:
        healthy = await self._broker.ahealth_check()
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.OK if healthy else ProbeStatus.FAILED,
            detail=None if healthy else "broker unreachable",
        )
