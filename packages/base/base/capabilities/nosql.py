from fastapi import FastAPI

from base.models.definitions.health import ProbeStatus
from base.models.view_models.health import ProbeResult
from base.services.nosql import AbstractNoSQLService


class NoSQLDependency:
    """Readiness for the service's own document database.

    The service is injected rather than resolved here, so `base` stays free of
    a hard Mongo or Cosmos import.
    """

    name = "nosql"

    def __init__(self, service: AbstractNoSQLService) -> None:
        self._service = service

    async def astartup(self, app: FastAPI) -> None:
        del app

    async def ashutdown(self, app: FastAPI) -> None:
        del app
        await self._service.adispose()

    async def aprobe(self) -> ProbeResult:
        healthy = await self._service.ahealth_check()
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.OK if healthy else ProbeStatus.FAILED,
            detail=None if healthy else "document database unreachable",
        )
