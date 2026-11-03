from collections.abc import Sequence
from types import ModuleType

from fastapi import FastAPI

from base.models.definitions.health import ProbeStatus
from base.models.view_models.health import ProbeResult
from base.services.prompt.loader import PromptLoader
from base.services.prompt.protocol import PromptTemplateStore


class Prompts:
    """Loads the service's prompt templates.

    Pass `store` to let stored templates override the ones in code; the
    registry then keeps itself current for as long as the service runs.
    """

    name = "prompts"

    def __init__(
        self,
        templates: Sequence[ModuleType] = (),
        store: PromptTemplateStore | None = None,
    ) -> None:
        self._templates = tuple(templates)
        self._store = store

    async def astartup(self, app: FastAPI) -> None:
        await PromptLoader().initialize(self._templates, self._store)

    async def ashutdown(self, app: FastAPI) -> None:
        await PromptLoader().shutdown()

    async def aprobe(self) -> ProbeResult:
        keys = PromptLoader().list_keys()
        if not keys:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.DEGRADED,
                detail="No prompts registered.",
            )
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.OK,
            detail=f"{len(keys)} prompts loaded.",
        )
