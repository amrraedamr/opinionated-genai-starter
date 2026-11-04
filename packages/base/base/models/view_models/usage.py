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
    cached_input_per_million: Decimal | None = None
    """Discounted rate for cache reads. Unset prices them as plain input."""

    cache_write_per_million: Decimal | None = None
    """Premium rate for cache writes. Unset prices them as plain input."""

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Decimal:
        """Price one call. Both cache counts are subsets of `input_tokens`."""
        total_input = max(input_tokens, 0)
        cached = min(max(cached_input_tokens, 0), total_input)
        written = min(max(cache_write_tokens, 0), total_input - cached)
        uncached = total_input - cached - written

        cached_rate = (
            self.input_per_million
            if self.cached_input_per_million is None
            else self.cached_input_per_million
        )
        write_rate = (
            self.input_per_million
            if self.cache_write_per_million is None
            else self.cache_write_per_million
        )
        return (
            self.input_per_million * Decimal(uncached)
            + cached_rate * Decimal(cached)
            + write_rate * Decimal(written)
            + self.output_per_million * Decimal(output_tokens)
        ) / Decimal(1_000_000)


class UsageRecord(BaseModel):
    """One model call, as billed."""

    model: str
    label: str
    """Who made the call — usually the agent class name."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    """Share of `input_tokens` read from the prompt cache, billed cheaper."""

    cache_write_tokens: int = 0
    """Share of `input_tokens` written to the prompt cache, billed dearer."""

    duration_ms: int = 0
    cost_usd: Decimal | None = None
    """None when no rate is known for the model, rather than a misleading zero."""

    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        """Input tokens neither read from nor written to the cache."""
        return max(
            self.input_tokens - self.cached_input_tokens - self.cache_write_tokens, 0
        )


class UsageTotals(BaseModel):
    """Everything one request spent."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
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
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache, between 0 and 1.

        Collapses when prompt-prefix stability breaks, before the bill shows it.
        """
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0

    @property
    def fully_priced(self) -> bool:
        """Whether `cost_usd` accounts for every call."""
        return self.priced_calls == self.calls
