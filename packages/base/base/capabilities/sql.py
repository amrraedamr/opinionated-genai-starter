from fastapi import FastAPI

from base.models.definitions.health import ProbeStatus
from base.models.view_models.health import ProbeResult
from base.services.sql.service import SQLService


class SQLDependency:
    """Readiness for the service's own SQL database."""

    name = "database"

    async def astartup(self, app: FastAPI) -> None:
        del app

    async def ashutdown(self, app: FastAPI) -> None:
        del app
        await SQLService().adispose()

    async def aprobe(self) -> ProbeResult:
        healthy = await SQLService().ahealth_check()
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.OK if healthy else ProbeStatus.FAILED,
            detail=None if healthy else "database unreachable",
        )
