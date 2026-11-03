"""Token and cost accounting for one request.

Every LLM call is recorded at the `OpenAIService` boundary, so an agent does
not have to remember to. Read the result at the end of a request:

    totals = UsageTracker().totals
    totals.total_tokens, totals.cost_usd

Rates are per million tokens in USD and change often, so they are
configuration rather than a constant: `UsageTracker.set_rates` replaces them,
and anything without a rate is counted in tokens but left out of the cost,
which `UsageTotals.fully_priced` reports.
"""

from decimal import Decimal

from base.models.view_models.usage import ModelRate, UsageRecord, UsageTotals
from base.services.common.meta import Stored

_DEFAULT_RATES: dict[str, ModelRate] = {
    "gpt-5.4": ModelRate(
        input_per_million=Decimal("1.25"), output_per_million=Decimal("10.00")
    ),
    "gpt-5.4-mini": ModelRate(
        input_per_million=Decimal("0.25"), output_per_million=Decimal("2.00")
    ),
    "gpt-5.4-nano": ModelRate(
        input_per_million=Decimal("0.05"), output_per_million=Decimal("0.40")
    ),
    "gpt-4o": ModelRate(
        input_per_million=Decimal("2.50"), output_per_million=Decimal("10.00")
    ),
    "gpt-4o-mini": ModelRate(
        input_per_million=Decimal("0.15"), output_per_million=Decimal("0.60")
    ),
    "text-embedding-3-small": ModelRate(
        input_per_million=Decimal("0.02"), output_per_million=Decimal("0")
    ),
    "text-embedding-3-large": ModelRate(
        input_per_million=Decimal("0.13"), output_per_million=Decimal("0")
    ),
}


class UsageTracker(metaclass=Stored):
    """Accumulates model usage for the current request.

    One instance per request scope, like `ExecutionContext`, so concurrent
    requests cannot bill each other. Outside a request scope it still works —
    a worker or a script gets one tracker for its scope.
    """

    _rates: dict[str, ModelRate] = dict(_DEFAULT_RATES)

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    @classmethod
    def set_rates(cls, rates: dict[str, ModelRate]) -> None:
        """Replace the rate table, for current prices or a private agreement."""
        cls._rates = dict(rates)

    @classmethod
    def rate_for(cls, model: str) -> ModelRate | None:
        """The rate for a model, matching an Azure deployment by its suffix.

        Deployments are named freely, so an exact miss falls back to the
        longest known model name the deployment ends with — `prod-gpt-4o`
        bills as `gpt-4o`.
        """
        exact = cls._rates.get(model)
        if exact is not None:
            return exact

        candidates = [name for name in cls._rates if model.endswith(name)]
        if not candidates:
            return None
        return cls._rates[max(candidates, key=len)]

    def record(
        self,
        model: str,
        label: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int = 0,
        error: str | None = None,
    ) -> UsageRecord:
        """Record one call. Returns what was recorded, priced if a rate exists."""
        rate = self.rate_for(model)
        entry = UsageRecord(
            model=model,
            label=label,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cost_usd=rate.cost(input_tokens, output_tokens) if rate else None,
            error=error,
        )
        self._records.append(entry)
        return entry

    @property
    def records(self) -> list[UsageRecord]:
        """Every call this request made, in order."""
        return list(self._records)

    @property
    def totals(self) -> UsageTotals:
        """What the request spent, across every call."""
        totals = UsageTotals()
        for entry in self._records:
            totals.calls += 1
            totals.input_tokens += entry.input_tokens
            totals.output_tokens += entry.output_tokens
            totals.duration_ms += entry.duration_ms
            totals.by_model[entry.model] = (
                totals.by_model.get(entry.model, 0) + entry.total_tokens
            )
            if entry.cost_usd is not None:
                totals.cost_usd += entry.cost_usd
                totals.priced_calls += 1
        return totals

    def reset(self) -> None:
        self._records.clear()
