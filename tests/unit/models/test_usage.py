"""Token counting and call pricing: `ProviderUsage`, `ModelRate`, `UsageRecord`.

Money is the reason these use `Decimal`, and the reason the cache counts are
clamped: a provider that reports more cached tokens than input tokens must not
produce a negative charge.
"""

from decimal import Decimal

from base.models.view_models.usage import (
    ModelRate,
    ProviderUsage,
    UsageRecord,
    UsageTotals,
)

RATE = ModelRate(
    input_per_million=Decimal("10"),
    output_per_million=Decimal("30"),
    cached_input_per_million=Decimal("1"),
    cache_write_per_million=Decimal("12.5"),
)

BARE_RATE = ModelRate(
    input_per_million=Decimal("10"),
    output_per_million=Decimal("30"),
)


def test_provider_usage_totals_only_input_and_output() -> None:
    usage = ProviderUsage(
        input_tokens=100,
        output_tokens=25,
        cached_input_tokens=60,
        cache_write_tokens=10,
    )
    assert usage.total_tokens == 125


def test_provider_usage_defaults_every_count_to_zero() -> None:
    usage = ProviderUsage()
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (0, 0, 0)
    assert (usage.cached_input_tokens, usage.cache_write_tokens) == (0, 0)


def test_both_cache_counts_are_subsets_of_the_input_count() -> None:
    """Pricing the three slices separately must not exceed pricing them as input."""
    usage = ProviderUsage(
        input_tokens=1_000, cached_input_tokens=600, cache_write_tokens=200
    )
    uncached = usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
    assert uncached == 200
    assert usage.cached_input_tokens + usage.cache_write_tokens <= usage.input_tokens


def test_a_call_with_no_caching_costs_input_plus_output() -> None:
    assert RATE.cost(1_000_000, 1_000_000) == Decimal("40")


def test_a_fully_cached_call_is_priced_at_the_cache_read_rate() -> None:
    assert RATE.cost(1_000_000, 0, cached_input_tokens=1_000_000) == Decimal("1")


def test_a_fully_written_call_is_priced_at_the_cache_write_rate() -> None:
    assert RATE.cost(1_000_000, 0, cache_write_tokens=1_000_000) == Decimal("12.5")


def test_a_mixed_call_prices_each_slice_at_its_own_rate() -> None:
    cost = RATE.cost(
        1_000_000, 0, cached_input_tokens=400_000, cache_write_tokens=100_000
    )
    expected = (
        Decimal("10") * 500_000 + Decimal("1") * 400_000 + Decimal("12.5") * 100_000
    ) / Decimal(1_000_000)
    assert cost == expected == Decimal("6.65")


def test_an_unset_cached_rate_prices_cache_reads_as_plain_input() -> None:
    assert BARE_RATE.cost(1_000, 0, cached_input_tokens=1_000) == BARE_RATE.cost(
        1_000, 0
    )


def test_an_unset_write_rate_prices_cache_writes_as_plain_input() -> None:
    assert BARE_RATE.cost(1_000, 0, cache_write_tokens=1_000) == BARE_RATE.cost(
        1_000, 0
    )


def test_a_cached_count_above_the_input_count_is_clamped_to_it() -> None:
    assert RATE.cost(100, 0, cached_input_tokens=1_000) == RATE.cost(
        100, 0, cached_input_tokens=100
    )


def test_a_write_count_is_clamped_to_what_the_cached_count_left_over() -> None:
    cost = RATE.cost(100, 0, cached_input_tokens=80, cache_write_tokens=80)
    expected = (Decimal("1") * 80 + Decimal("12.5") * 20) / Decimal(1_000_000)
    assert cost == expected


def test_a_negative_input_count_costs_nothing_rather_than_a_credit() -> None:
    assert RATE.cost(-5, 0) == Decimal(0)


def test_a_negative_cache_count_is_treated_as_zero() -> None:
    assert RATE.cost(100, 0, cached_input_tokens=-5) == RATE.cost(100, 0)


def test_cost_stays_a_decimal_so_cents_never_drift() -> None:
    assert isinstance(RATE.cost(3, 7), Decimal)
    assert RATE.cost(1, 0) == Decimal("10") / Decimal(1_000_000)


def test_a_usage_record_identifies_itself_and_defaults_to_one_attempt() -> None:
    record = UsageRecord(model="gpt-5", label="Agent")
    assert record.attempts == 1
    assert record.retried is False
    assert record.attempt_errors == []
    assert record.cost_usd is None
    assert record.call_id != UsageRecord(model="gpt-5", label="Agent").call_id


def test_a_record_that_took_more_than_one_try_says_it_was_retried() -> None:
    record = UsageRecord(
        model="gpt-5", label="Agent", attempts=3, attempt_errors=["a", "b"]
    )
    assert record.retried is True
    assert record.attempt_errors == ["a", "b"]


def test_a_records_uncached_input_is_what_neither_cache_slice_claimed() -> None:
    record = UsageRecord(
        model="gpt-5",
        label="Agent",
        input_tokens=1_000,
        cached_input_tokens=600,
        cache_write_tokens=100,
    )
    assert record.uncached_input_tokens == 300


def test_uncached_input_never_goes_negative_when_counts_disagree() -> None:
    record = UsageRecord(
        model="gpt-5",
        label="Agent",
        input_tokens=100,
        cached_input_tokens=90,
        cache_write_tokens=90,
    )
    assert record.uncached_input_tokens == 0


def test_a_record_totals_only_input_and_output() -> None:
    record = UsageRecord(
        model="gpt-5", label="Agent", input_tokens=40, output_tokens=2, duration_ms=12
    )
    assert record.total_tokens == 42


def test_totals_report_the_cache_hit_rate_as_a_share_of_input() -> None:
    totals = UsageTotals(input_tokens=1_000, cached_input_tokens=250)
    assert totals.cache_hit_rate == 0.25


def test_totals_report_a_zero_hit_rate_rather_than_dividing_by_zero() -> None:
    assert UsageTotals().cache_hit_rate == 0.0


def test_totals_say_whether_every_call_had_a_known_rate() -> None:
    assert UsageTotals(calls=3, priced_calls=3).fully_priced is True
    assert UsageTotals(calls=3, priced_calls=2).fully_priced is False


def test_totals_start_at_zero_cost_rather_than_none() -> None:
    totals = UsageTotals()
    assert totals.cost_usd == Decimal(0)
    assert totals.by_model == {}
