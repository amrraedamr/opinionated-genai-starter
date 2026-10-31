"""Token and cost accounting per request."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from base.models.view_models.usage import ModelRate
from base.services.common.meta import Stored, scope_id
from base.services.usage import UsageTracker


@pytest.fixture(autouse=True)
def fresh_scope():
    """Each test gets its own request scope, as a real request would."""
    scope_id.set(uuid4())
    yield
    Stored.delete_scope()


class TestRates:
    def test_a_known_model_is_priced(self):
        rate = UsageTracker.rate_for("gpt-4o")
        assert rate is not None
        assert rate.input_per_million > 0

    def test_an_azure_deployment_bills_as_the_model_it_runs(self):
        assert UsageTracker.rate_for("prod-gpt-4o") == UsageTracker.rate_for("gpt-4o")

    def test_the_longest_match_wins(self):
        assert UsageTracker.rate_for("prod-gpt-4o-mini") == UsageTracker.rate_for(
            "gpt-4o-mini"
        )

    def test_an_unknown_model_has_no_rate(self):
        assert UsageTracker.rate_for("llama-anything") is None

    def test_cost_is_per_million_tokens_each_way(self):
        rate = ModelRate(
            input_per_million=Decimal("2.00"), output_per_million=Decimal("10.00")
        )
        assert rate.cost(1_000_000, 0) == Decimal("2.00")
        assert rate.cost(0, 1_000_000) == Decimal("10.00")
        assert rate.cost(500_000, 100_000) == Decimal("2.00")


class TestRecording:
    def test_a_priced_call_carries_its_cost(self):
        tracker = UsageTracker()
        entry = tracker.record("gpt-4o", "Agent", input_tokens=1000, output_tokens=500)

        assert entry.total_tokens == 1500
        assert entry.cost_usd is not None
        assert entry.cost_usd > 0

    def test_an_unpriced_call_has_no_cost_rather_than_zero(self):
        entry = UsageTracker().record("llama-anything", "Agent", 1000, 500)
        assert entry.cost_usd is None

    def test_totals_add_up_across_calls(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", "A", 100, 50, duration_ms=10)
        tracker.record("gpt-4o", "B", 200, 80, duration_ms=20)

        totals = tracker.totals
        assert totals.calls == 2
        assert totals.input_tokens == 300
        assert totals.output_tokens == 130
        assert totals.total_tokens == 430
        assert totals.duration_ms == 30
        assert totals.cost_usd > 0
        assert totals.fully_priced

    def test_tokens_are_broken_down_by_model(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", "A", 100, 50)
        tracker.record("gpt-4o-mini", "B", 10, 5)

        assert tracker.totals.by_model == {"gpt-4o": 150, "gpt-4o-mini": 15}

    def test_an_unpriced_call_makes_the_cost_a_floor(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", "A", 100, 50)
        tracker.record("llama-anything", "B", 100, 50)

        totals = tracker.totals
        assert totals.calls == 2
        assert totals.priced_calls == 1
        assert not totals.fully_priced
        assert totals.total_tokens == 300

    def test_a_failed_call_still_counts(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", "A", 100, 0, error="rate limited")

        assert tracker.totals.calls == 1
        assert tracker.records[0].error == "rate limited"

    def test_records_come_back_in_order(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", "first", 1, 1)
        tracker.record("gpt-4o", "second", 1, 1)

        assert [r.label for r in tracker.records] == ["first", "second"]

    def test_reset_clears_the_request(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", "A", 100, 50)
        tracker.reset()

        assert tracker.totals.calls == 0


class TestScoping:
    def test_one_tracker_per_request(self):
        UsageTracker().record("gpt-4o", "A", 100, 50)
        assert UsageTracker().totals.calls == 1

    def test_a_different_request_bills_separately(self):
        UsageTracker().record("gpt-4o", "A", 100, 50)

        scope_id.set(uuid4())
        assert UsageTracker().totals.calls == 0

    def test_custom_rates_replace_the_defaults(self):
        original = UsageTracker._rates
        try:
            UsageTracker.set_rates(
                {
                    "my-model": ModelRate(
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("1"),
                    )
                }
            )
            assert UsageTracker.rate_for("gpt-4o") is None
            assert UsageTracker.rate_for("my-model") is not None
        finally:
            UsageTracker._rates = original


class TestLLMServiceIntegration:
    def test_every_model_call_is_recorded_without_the_agent_asking(self):
        from openai.types.completion_usage import CompletionUsage

        from base.services.llm import OpenAIService

        service = OpenAIService()
        service._record_call(
            label="SummaryAgent",
            prompt=[],
            response="done",
            usage=CompletionUsage(
                prompt_tokens=120, completion_tokens=30, total_tokens=150
            ),
            duration_ms=42,
            error=None,
        )

        records = UsageTracker().records
        assert len(records) == 1
        assert records[0].label == "SummaryAgent"
        assert records[0].input_tokens == 120
        assert records[0].output_tokens == 30
        assert records[0].duration_ms == 42

    def test_a_failed_call_is_recorded_too(self):
        from base.services.llm import OpenAIService

        OpenAIService()._record_call(
            label="SummaryAgent",
            prompt=[],
            response=None,
            usage=None,
            duration_ms=5,
            error="RateLimitError: slow down",
        )

        assert UsageTracker().records[0].error == "RateLimitError: slow down"
