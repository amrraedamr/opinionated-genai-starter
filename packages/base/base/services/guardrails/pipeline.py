"""Running several guardrails over one piece of content, in order.

The pipeline sits beside the seam rather than in it, the way a backend sits
beside the abstraction it implements: `AbstractGuardrail` is the contract, and
this is one way of composing several of them.
"""

from collections.abc import Sequence

from base.models.definitions.guardrails import GuardrailOutcome, GuardrailStage
from base.models.view_models.guardrails import GuardrailReport, GuardrailVerdict
from base.services.guardrails import AbstractGuardrail, GuardrailBlocked


class GuardrailPipeline:
    """Guardrails run in order. The first block wins; redactions compose."""

    def __init__(self, *guardrails: AbstractGuardrail) -> None:
        names = [g.name for g in guardrails]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate guardrail names: {duplicates}")
        self._guardrails: tuple[AbstractGuardrail, ...] = guardrails

    @property
    def guardrails(self) -> tuple[AbstractGuardrail, ...]:
        return self._guardrails

    def applies_to(self, stage: GuardrailStage) -> bool:
        """Whether any guardrail here checks `stage`."""
        return any(stage in g.stages for g in self._guardrails)

    def run(self, content: str, stage: GuardrailStage) -> GuardrailReport:
        """Run every guardrail checking `stage` over `content`, in order."""
        verdicts: list[GuardrailVerdict] = []
        current = content
        for guardrail in self._for(stage):
            verdict = self._named(guardrail.check(current, stage), guardrail)
            verdicts.append(verdict)
            if verdict.outcome == GuardrailOutcome.BLOCK:
                return self._report(stage, GuardrailOutcome.BLOCK, current, verdicts)
            if verdict.outcome == GuardrailOutcome.REDACT:
                current = verdict.content or ""
        return self._report(stage, self._settled(verdicts), current, verdicts)

    async def arun(self, content: str, stage: GuardrailStage) -> GuardrailReport:
        """Run every guardrail checking `stage` over `content`, in order, asynchronously."""
        verdicts: list[GuardrailVerdict] = []
        current = content
        for guardrail in self._for(stage):
            verdict = self._named(await guardrail.acheck(current, stage), guardrail)
            verdicts.append(verdict)
            if verdict.outcome == GuardrailOutcome.BLOCK:
                return self._report(stage, GuardrailOutcome.BLOCK, current, verdicts)
            if verdict.outcome == GuardrailOutcome.REDACT:
                current = verdict.content or ""
        return self._report(stage, self._settled(verdicts), current, verdicts)

    def enforce(self, content: str, stage: GuardrailStage) -> str:
        """The content that may proceed, or `GuardrailBlocked` when none may."""
        report = self.run(content, stage)
        if report.blocked:
            raise GuardrailBlocked(report)
        return report.content

    async def aenforce(self, content: str, stage: GuardrailStage) -> str:
        """The content that may proceed, or `GuardrailBlocked` when none may, asynchronously."""
        report = await self.arun(content, stage)
        if report.blocked:
            raise GuardrailBlocked(report)
        return report.content

    def _for(self, stage: GuardrailStage) -> Sequence[AbstractGuardrail]:
        return [g for g in self._guardrails if stage in g.stages]

    @staticmethod
    def _named(
        verdict: GuardrailVerdict, guardrail: AbstractGuardrail
    ) -> GuardrailVerdict:
        """Stamp which guardrail reached the verdict, so a report can say."""
        if verdict.guardrail is not None:
            return verdict
        return verdict.model_copy(update={"guardrail": guardrail.name})

    @staticmethod
    def _settled(verdicts: Sequence[GuardrailVerdict]) -> GuardrailOutcome:
        if any(v.outcome == GuardrailOutcome.REDACT for v in verdicts):
            return GuardrailOutcome.REDACT
        return GuardrailOutcome.ALLOW

    @staticmethod
    def _report(
        stage: GuardrailStage,
        outcome: GuardrailOutcome,
        content: str,
        verdicts: list[GuardrailVerdict],
    ) -> GuardrailReport:
        return GuardrailReport(
            stage=stage, outcome=outcome, content=content, verdicts=verdicts
        )


__all__ = ["GuardrailPipeline"]
