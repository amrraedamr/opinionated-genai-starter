"""What a guardrail decided, and what a pipeline of them decided together."""

from typing import Self

from pydantic import Field, model_validator

from base.models.definitions.guardrails import GuardrailOutcome, GuardrailStage
from base.models.view_models import BaseModel


class GuardrailVerdict(BaseModel):
    """One guardrail's decision about one piece of content."""

    outcome: GuardrailOutcome
    content: str | None = None
    reason: str | None = None
    category: str | None = None
    guardrail: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.outcome == GuardrailOutcome.REDACT and self.content is None:
            raise ValueError("A redaction must carry the content that replaces it.")
        if self.outcome == GuardrailOutcome.BLOCK and not self.reason:
            raise ValueError("A block must say why.")
        return self

    @classmethod
    def allow(cls) -> Self:
        """Nothing to object to."""
        return cls(outcome=GuardrailOutcome.ALLOW)

    @classmethod
    def redact(cls, content: str, *, category: str | None = None) -> Self:
        """Let it through, with `content` in place of what was shown."""
        return cls(outcome=GuardrailOutcome.REDACT, content=content, category=category)

    @classmethod
    def block(cls, reason: str, *, category: str | None = None) -> Self:
        """Refuse it, saying why."""
        return cls(outcome=GuardrailOutcome.BLOCK, reason=reason, category=category)


class GuardrailReport(BaseModel):
    """Every verdict a stage reached, and the content that survives them."""

    stage: GuardrailStage
    outcome: GuardrailOutcome
    content: str
    verdicts: list[GuardrailVerdict] = Field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.outcome == GuardrailOutcome.BLOCK

    @property
    def blocking_verdict(self) -> GuardrailVerdict | None:
        """The verdict that refused the content, when one did."""
        return next(
            (v for v in self.verdicts if v.outcome == GuardrailOutcome.BLOCK), None
        )
