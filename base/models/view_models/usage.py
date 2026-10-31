"""What a model call cost, in tokens and in money."""

from decimal import Decimal

from pydantic import Field

from base.models.view_models import BaseModel


class ModelRate(BaseModel):
    """USD per million tokens, split by direction.

    Decimal rather than float: these are multiplied by token counts and summed
    across a request, and a cent that drifts is a cent nobody can reconcile.
    """

    input_per_million: Decimal
    output_per_million: Decimal

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            self.input_per_million * Decimal(input_tokens)
            + self.output_per_million * Decimal(output_tokens)
        ) / Decimal(1_000_000)


class UsageRecord(BaseModel):
    """One model call, as billed."""

    model: str
    label: str
    """Who made the call — usually the agent class name."""

    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    cost_usd: Decimal | None = None
    """None when no rate is known for the model, rather than a misleading zero."""

    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageTotals(BaseModel):
    """Everything one request spent."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    cost_usd: Decimal = Decimal(0)
    priced_calls: int = 0
    """Calls that had a known rate. Below `calls` means `cost_usd` is a floor."""

    by_model: dict[str, int] = Field(default_factory=dict)
    """Total tokens per model, for seeing which one is actually expensive."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def fully_priced(self) -> bool:
        """Whether `cost_usd` accounts for every call."""
        return self.priced_calls == self.calls
