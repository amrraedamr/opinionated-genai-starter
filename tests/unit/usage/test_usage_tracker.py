"""`UsageTracker` is the ledger one request is billed against.

Money is the reason this is `Decimal` and the reason an unknown model records
tokens but no cost: a zero there would read as "free" rather than "unpriced",
and `fully_priced` is how a caller tells the two apart.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from decimal import Decimal

import pytest

from base.models.view_models.usage import ModelRate
from base.services.common.meta import Stored, scope_id
from base.services.usage import _DEFAULT_RATES, UsageTracker
from base.tools.attempt_trail import mark_attempt, mark_attempt_failure


@pytest.fixture(autouse=True)
def _own_scope() -> Iterator[None]:
    """One tracker per test, the way one tracker serves one request."""
    token = scope_id.set(uuid.uuid4())
    original = UsageTracker._rates
    yield
    UsageTracker._rates = original
    Stored.delete_scope()
    scope_id.reset(token)


def test_a_fresh_tracker_has_spent_nothing() -> None:
    totals = UsageTracker().totals

    assert totals.calls == 0
    assert totals.total_tokens == 0
    assert totals.cost_usd == Decimal(0)
    assert totals.by_model == {}
    assert totals.fully_priced is True


def test_one_call_is_recorded_with_what_it_consumed() -> None:
    record = UsageTracker().record("gpt-4o", "Answerer", 1000, 500, duration_ms=120)

    assert record.model == "gpt-4o"
    assert record.label == "Answerer"
    assert record.input_tokens == 1000
    assert record.output_tokens == 500
    assert record.total_tokens == 1500
    assert record.duration_ms == 120
    assert record.error is None


def test_calls_accumulate_in_the_order_they_were_made() -> None:
    tracker = UsageTracker()
    tracker.record("gpt-4o", "First", 10, 1)
    tracker.record("gpt-4o-mini", "Second", 20, 2)

    assert [r.label for r in tracker.records] == ["First", "Second"]
    assert tracker.totals.calls == 2
    assert tracker.totals.input_tokens == 30
    assert tracker.totals.output_tokens == 3


def test_tokens_are_aggregated_per_model() -> None:
    tracker = UsageTracker()
    tracker.record("gpt-4o", "A", 100, 10)
    tracker.record("gpt-4o", "B", 200, 20)
    tracker.record("gpt-4o-mini", "C", 50, 5)

    assert tracker.totals.by_model == {"gpt-4o": 330, "gpt-4o-mini": 55}


def test_each_label_keeps_its_own_record_so_an_agent_can_be_attributed() -> None:
    tracker = UsageTracker()
    tracker.record("gpt-4o", "Router", 100, 10)
    tracker.record("gpt-4o", "Answerer", 900, 90)
    tracker.record("gpt-4o", "Answerer", 100, 10)

    by_label: dict[str, int] = {}
    for record in tracker.records:
        by_label[record.label] = by_label.get(record.label, 0) + record.total_tokens

    assert by_label == {"Router": 110, "Answerer": 1100}


def test_cost_is_the_rate_applied_to_each_direction() -> None:
    UsageTracker.set_rates(
        {
            "m": ModelRate(
                input_per_million=Decimal("2.00"), output_per_million=Decimal("10.00")
            )
        }
    )

    record = UsageTracker().record("m", "A", 1_000_000, 500_000)

    assert record.cost_usd == Decimal("7.00")


def test_costs_sum_across_the_request() -> None:
    UsageTracker.set_rates(
        {
            "m": ModelRate(
                input_per_million=Decimal("1.00"), output_per_million=Decimal("2.00")
            )
        }
    )
    tracker = UsageTracker()
    tracker.record("m", "A", 1_000_000, 1_000_000)
    tracker.record("m", "B", 500_000, 0)

    assert tracker.totals.cost_usd == Decimal("3.50")
    assert tracker.totals.priced_calls == 2
    assert tracker.totals.fully_priced is True


def test_a_model_with_no_rate_is_counted_but_not_priced() -> None:
    tracker = UsageTracker()
    record = tracker.record("a-model-nobody-priced", "A", 1000, 100)

    assert record.cost_usd is None
    assert tracker.totals.by_model == {"a-model-nobody-priced": 1100}
    assert tracker.totals.cost_usd == Decimal(0)
    assert tracker.totals.fully_priced is False


def test_an_azure_deployment_bills_as_the_model_its_name_ends_with() -> None:
    assert UsageTracker.rate_for("prod-gpt-4o") == _DEFAULT_RATES["gpt-4o"]


def test_the_longest_matching_model_name_wins() -> None:
    assert UsageTracker.rate_for("eu-gpt-4o-mini") == _DEFAULT_RATES["gpt-4o-mini"]


def test_an_exact_match_beats_a_suffix_match() -> None:
    assert UsageTracker.rate_for("gpt-4o") == _DEFAULT_RATES["gpt-4o"]


def test_a_name_resembling_nothing_known_has_no_rate() -> None:
    assert UsageTracker.rate_for("llama-3-70b") is None


def test_replacing_the_rate_table_replaces_it_wholesale() -> None:
    UsageTracker.set_rates(
        {
            "private": ModelRate(
                input_per_million=Decimal("0.10"), output_per_million=Decimal("0.20")
            )
        }
    )

    assert UsageTracker.rate_for("private") is not None
    assert UsageTracker.rate_for("gpt-4o") is None


def test_cached_input_tokens_are_a_share_of_the_input_not_an_addition() -> None:
    UsageTracker.set_rates(
        {
            "m": ModelRate(
                input_per_million=Decimal("10.00"),
                output_per_million=Decimal("0"),
                cached_input_per_million=Decimal("1.00"),
            )
        }
    )

    record = UsageTracker().record(
        "m", "A", input_tokens=1_000_000, output_tokens=0, cached_input_tokens=900_000
    )

    assert record.cost_usd == Decimal("1.90")
    assert record.uncached_input_tokens == 100_000


def test_cache_writes_are_billed_at_their_own_rate() -> None:
    UsageTracker.set_rates(
        {
            "m": ModelRate(
                input_per_million=Decimal("10.00"),
                output_per_million=Decimal("0"),
                cache_write_per_million=Decimal("12.50"),
            )
        }
    )

    record = UsageTracker().record(
        "m", "A", input_tokens=1_000_000, output_tokens=0, cache_write_tokens=400_000
    )

    assert record.cost_usd == Decimal("11.00")


def test_cache_counts_reach_the_totals_and_the_hit_rate() -> None:
    tracker = UsageTracker()
    tracker.record(
        "gpt-4o",
        "A",
        input_tokens=1000,
        output_tokens=0,
        cached_input_tokens=750,
        cache_write_tokens=100,
    )

    totals = tracker.totals
    assert totals.cached_input_tokens == 750
    assert totals.cache_write_tokens == 100
    assert totals.cache_hit_rate == 0.75


def test_a_failed_call_is_recorded_with_why() -> None:
    record = UsageTracker().record("gpt-4o", "A", 10, 0, error="LLMTimeoutError: slow")

    assert record.error == "LLMTimeoutError: slow"
    assert UsageTracker().totals.calls == 1


def test_a_call_id_can_be_supplied_so_it_matches_its_debug_call() -> None:
    call_id = uuid.uuid4()

    record = UsageTracker().record("gpt-4o", "A", 10, 1, call_id=call_id)

    assert record.call_id == call_id


def test_a_call_without_an_id_still_gets_a_unique_one() -> None:
    tracker = UsageTracker()
    first = tracker.record("gpt-4o", "A", 1, 1)
    second = tracker.record("gpt-4o", "A", 1, 1)

    assert first.call_id != second.call_id


def test_a_call_that_succeeded_first_time_reports_one_attempt() -> None:
    record = UsageTracker().record("gpt-4o", "A", 1, 1)

    assert record.attempts == 1
    assert record.retried is False
    assert record.attempt_errors == []


def test_a_retried_call_is_one_record_carrying_what_it_recovered_from() -> None:
    mark_attempt()
    mark_attempt_failure(TimeoutError("too slow"))
    mark_attempt()

    record = UsageTracker().record("gpt-4o", "A", 10, 1)

    assert record.attempts == 2
    assert record.retried is True
    assert record.attempt_errors == ["TimeoutError: too slow"]
    assert UsageTracker().totals.calls == 1


def test_the_attempt_trail_is_not_inherited_by_the_next_call() -> None:
    mark_attempt()
    mark_attempt()
    tracker = UsageTracker()
    tracker.record("gpt-4o", "A", 1, 1)

    assert tracker.record("gpt-4o", "B", 1, 1).attempts == 1


def test_reset_clears_the_ledger() -> None:
    tracker = UsageTracker()
    tracker.record("gpt-4o", "A", 10, 1)

    tracker.reset()

    assert tracker.records == []
    assert tracker.totals.calls == 0


def test_the_records_list_is_a_copy_callers_cannot_corrupt() -> None:
    tracker = UsageTracker()
    tracker.record("gpt-4o", "A", 10, 1)

    tracker.records.clear()

    assert len(tracker.records) == 1


def test_one_scope_sees_one_tracker() -> None:
    assert UsageTracker() is UsageTracker()


def test_another_scope_bills_separately() -> None:
    UsageTracker().record("gpt-4o", "A", 10, 1)

    token = scope_id.set(uuid.uuid4())
    try:
        assert UsageTracker().totals.calls == 0
    finally:
        Stored.delete_scope()
        scope_id.reset(token)

    assert UsageTracker().totals.calls == 1


def test_durations_sum_across_the_request() -> None:
    tracker = UsageTracker()
    tracker.record("gpt-4o", "A", 1, 1, duration_ms=100)
    tracker.record("gpt-4o", "B", 1, 1, duration_ms=250)

    assert tracker.totals.duration_ms == 350


def test_an_empty_request_has_no_cache_hit_rate_rather_than_dividing_by_zero() -> None:
    assert UsageTracker().totals.cache_hit_rate == 0.0
