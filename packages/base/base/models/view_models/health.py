from pydantic import Field

from base.models.definitions.health import ProbeStatus
from base.models.view_models import BaseModel


class ProbeResult(BaseModel):
    name: str
    status: ProbeStatus
    detail: str | None = None


class HealthReport(BaseModel):
    status: ProbeStatus
    checks: list[ProbeResult] = Field(default_factory=list)

    @classmethod
    def aggregate(cls, checks: list[ProbeResult]) -> "HealthReport":
        """Worst individual probe wins."""
        statuses = {check.status for check in checks}
        if ProbeStatus.FAILED in statuses:
            status = ProbeStatus.FAILED
        elif ProbeStatus.DEGRADED in statuses:
            status = ProbeStatus.DEGRADED
        else:
            status = ProbeStatus.OK
        return cls(status=status, checks=checks)
