"""Checking what goes into a model, what comes out, and what tools hand back.

A guardrail allows, redacts or blocks one piece of content at one stage. Tool
results get their own stage because a tool, above all a third-party MCP server,
can return text written to steer the model. `pipeline.GuardrailPipeline` runs
several of these in order.
"""

from abc import ABC, abstractmethod
from typing import final

from base.models.definitions.errors import GuardrailErrors
from base.models.definitions.guardrails import GuardrailStage
from base.models.view_models.guardrails import GuardrailReport, GuardrailVerdict
from base.tools.errors import ErrorsException


class GuardrailBlocked(ErrorsException):
    """Content was refused. The report says which guardrail and why."""

    def __init__(self, report: GuardrailReport) -> None:
        super().__init__(error_enum=GuardrailErrors.CONTENT_BLOCKED)
        self.report = report


class AbstractGuardrail(ABC):
    """One check on content at the stages it declares."""

    @abstractmethod
    def _name(self) -> str: ...

    @final
    @property
    def name(self) -> str:
        return self._name()

    @abstractmethod
    def _stages(self) -> frozenset[GuardrailStage]: ...

    @final
    @property
    def stages(self) -> frozenset[GuardrailStage]:
        """The stages this guardrail checks. Others pass it by."""
        return self._stages()

    @abstractmethod
    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict: ...

    @final
    def check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        """Decide about `content` at `stage`."""
        return self._check(content, stage)

    @abstractmethod
    async def _acheck(
        self, content: str, stage: GuardrailStage
    ) -> GuardrailVerdict: ...

    @final
    async def acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        """Decide about `content` at `stage`, asynchronously."""
        return await self._acheck(content, stage)


__all__ = [
    "AbstractGuardrail",
    "GuardrailBlocked",
]
