from typing import Protocol, runtime_checkable

from base.tools.errors import MissingExtraError

try:
    from fastapi import FastAPI
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("server", "fastapi", feature="Capabilities") from exc

from base.models.definitions.health import ProbeStatus
from base.models.view_models.health import ProbeResult


@runtime_checkable
class Capability(Protocol):
    """A subsystem a service opts into.

    Build nothing expensive in `__init__`: capabilities are constructed at
    import, before gunicorn forks its workers. Create engines, pools and
    clients in `startup()`, which runs per worker after the fork.
    """

    name: str

    async def astartup(self, app: FastAPI) -> None: ...

    async def ashutdown(self, app: FastAPI) -> None: ...

    async def aprobe(self) -> ProbeResult: ...


__all__ = ["Capability", "ProbeResult", "ProbeStatus"]
