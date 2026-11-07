"""Pins the direction each provider's cache counts point.

`LLMUsageCounts` treats both cache figures as a *share* of `input_tokens`, so
cost maths is comparable across providers. Only Gemini and OpenAI report them
that way. Anthropic reports them alongside the input total instead, so its
adapter adds them in — and a test that does not name that asymmetry out loud
would let someone "simplify" the three into one and silently undercount every
Anthropic call.

Reasoning tokens point the other way: Gemini reports them beside the output
total and the adapter adds them in, while OpenAI's are already inside it.
"""

from typing import Any

import pytest
from anthropic.types import Message, TextBlock, Usage
from google.genai import types as genai_types
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)

from base.services.llm.anthropic import AnthropicService
from base.services.llm.gemini import GeminiService
from base.services.llm.openai import OpenAIService


def anthropic_usage(**counts: int) -> Message:
    return Message(
        id="msg_1",
        content=[TextBlock(type="text", text="x")],
        model="claude-test",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(**{"input_tokens": 0, "output_tokens": 0, **counts}),
    )


def gemini_usage(**counts: int) -> Any:
    return genai_types.GenerateContentResponse(
        candidates=[],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(**counts),
    )


def test_anthropic_cache_counts_are_additive_to_its_input_total() -> None:
    reported_input = 100
    counts = AnthropicService._to_usage_counts(
        anthropic_usage(
            input_tokens=reported_input,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=50,
        )
    )

    assert counts.input_tokens == reported_input + 30 + 50


def test_gemini_cache_counts_are_a_subset_of_its_input_total() -> None:
    reported_input = 100
    counts = GeminiService._to_usage_counts(
        gemini_usage(prompt_token_count=reported_input, cached_content_token_count=40)
    )

    assert counts.input_tokens == reported_input


def test_openai_cache_counts_are_a_subset_of_its_input_total() -> None:
    reported_input = 100
    counts = object.__new__(OpenAIService)._to_usage_counts(
        CompletionUsage(
            prompt_tokens=reported_input,
            completion_tokens=0,
            total_tokens=reported_input,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=40),
        )
    )

    assert counts is not None
    assert counts.input_tokens == reported_input


def test_the_same_billable_call_counts_the_same_through_all_three() -> None:
    """70 fresh input tokens, 30 cached, 20 output — however it was reported."""
    anthropic = AnthropicService._to_usage_counts(
        anthropic_usage(input_tokens=70, output_tokens=20, cache_read_input_tokens=30)
    )
    gemini = GeminiService._to_usage_counts(
        gemini_usage(
            prompt_token_count=100,
            candidates_token_count=20,
            cached_content_token_count=30,
        )
    )
    openai = object.__new__(OpenAIService)._to_usage_counts(
        CompletionUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=30),
        )
    )

    assert openai is not None
    for counts in (anthropic, gemini, openai):
        assert (counts.input_tokens, counts.output_tokens) == (100, 20)
        assert counts.cached_input_tokens == 30


@pytest.mark.parametrize(
    "counts",
    [
        AnthropicService._to_usage_counts(
            anthropic_usage(input_tokens=70, cache_read_input_tokens=30)
        ),
        GeminiService._to_usage_counts(
            gemini_usage(prompt_token_count=100, cached_content_token_count=30)
        ),
    ],
    ids=["anthropic", "gemini"],
)
def test_no_provider_reports_more_cached_tokens_than_input_tokens(
    counts: Any,
) -> None:
    assert counts.cached_input_tokens <= counts.input_tokens


def test_gemini_reasoning_tokens_are_additive_to_its_output_total() -> None:
    reported_output = 20
    counts = GeminiService._to_usage_counts(
        gemini_usage(candidates_token_count=reported_output, thoughts_token_count=7)
    )

    assert counts.output_tokens == reported_output + 7
    assert counts.reasoning_tokens == 7


def test_openai_reasoning_tokens_are_a_subset_of_its_output_total() -> None:
    reported_output = 20
    counts = object.__new__(OpenAIService)._to_usage_counts(
        CompletionUsage(
            prompt_tokens=0,
            completion_tokens=reported_output,
            total_tokens=reported_output,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=7),
        )
    )

    assert counts is not None
    assert counts.output_tokens == reported_output
    assert counts.reasoning_tokens == 7


def test_only_anthropic_and_openai_report_a_cache_write() -> None:
    anthropic = AnthropicService._to_usage_counts(
        anthropic_usage(cache_creation_input_tokens=50)
    )
    gemini = GeminiService._to_usage_counts(
        gemini_usage(prompt_token_count=100, cached_content_token_count=40)
    )

    assert anthropic.cache_write_tokens == 50
    assert gemini.cache_write_tokens == 0


def test_the_neutral_total_is_input_plus_output_for_every_provider() -> None:
    anthropic = AnthropicService._to_usage_counts(
        anthropic_usage(input_tokens=70, output_tokens=20, cache_read_input_tokens=30)
    )
    gemini = GeminiService._to_usage_counts(
        gemini_usage(prompt_token_count=100, candidates_token_count=20)
    )

    assert anthropic.total_tokens == 120
    assert gemini.total_tokens == 120
