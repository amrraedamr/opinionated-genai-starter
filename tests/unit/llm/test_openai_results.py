"""Pins how either OpenAI reply shape becomes the neutral result.

Both surfaces are folded into one `LLMResult`, which means two different usage
shapes and two different ways of saying why generation stopped. The Responses
API reports a status rather than a finish reason, and a reply cut short is
raised before it reaches the normaliser at all.
"""

from typing import Any

import pytest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from base.models.definitions.model import FinishReason
from base.models.view_models.exceptions.llm import (
    LLMContentFilterError,
    LLMIncompleteError,
)
from base.models.view_models.openai_results import CompletionResult, ResponseResult
from base.services.llm.openai import OpenAIService


def service() -> OpenAIService:
    """A service with only what the result methods read."""
    bare = object.__new__(OpenAIService)
    bare._model = "gpt-test"
    return bare


def response(**overrides: Any) -> Response:
    """A completed Responses-API reply, with nothing in it by default."""
    fields: dict[str, Any] = {
        "id": "resp_1",
        "created_at": 0,
        "model": "gpt-test",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
    }
    fields.update(overrides)
    return Response(**fields)


def text_item(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def completion_usage(**overrides: Any) -> CompletionUsage:
    fields: dict[str, Any] = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    fields.update(overrides)
    return CompletionUsage(**fields)


def response_usage(**overrides: Any) -> ResponseUsage:
    """The SDK makes both detail blocks mandatory, so they default to zero."""
    fields: dict[str, Any] = {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "input_tokens_details": InputTokensDetails(
            cached_tokens=0, cache_write_tokens=0
        ),
        "output_tokens_details": OutputTokensDetails(reasoning_tokens=0),
    }
    fields.update(overrides)
    return ResponseUsage(**fields)


def test_a_completion_becomes_the_neutral_result() -> None:
    result = service()._to_llm_result(
        CompletionResult(
            message=ChatCompletionMessage(role="assistant", content="Paris"),
            usage=completion_usage(),
            finish_reason="stop",
        )
    )

    assert result.content == "Paris"
    assert result.tool_calls is None
    assert result.reasoning is None
    assert result.finish_reason is FinishReason.STOP


def test_a_response_becomes_the_neutral_result_with_its_reasoning() -> None:
    result = service()._to_llm_result(
        ResponseResult(
            response=response(
                output=[
                    ResponseReasoningItem(
                        id="r1",
                        type="reasoning",
                        summary=[Summary(type="summary_text", text="thought")],
                    ),
                    text_item("Paris"),
                ]
            ),
            usage=response_usage(),
        )
    )

    assert result.content == "Paris"
    assert result.reasoning == "thought"
    assert result.finish_reason is FinishReason.STOP


def test_a_completion_carries_its_tool_calls_across() -> None:
    result = service()._to_llm_result(
        CompletionResult(
            message=ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id="call_1",
                        type="function",
                        function=Function(
                            name="get_weather", arguments='{"city": "Paris"}'
                        ),
                    )
                ],
            ),
            finish_reason="tool_calls",
        )
    )

    assert result.tool_calls is not None
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.finish_reason is FinishReason.TOOL_CALLS


def test_a_response_asking_for_a_tool_finishes_as_tool_calls() -> None:
    result = service()._to_llm_result(
        ResponseResult(
            response=response(
                output=[
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="call_1",
                        name="get_weather",
                        arguments="{}",
                    )
                ]
            )
        )
    )

    assert result.finish_reason is FinishReason.TOOL_CALLS


def test_a_response_that_answered_finishes_as_stop() -> None:
    result = service()._to_llm_result(
        ResponseResult(response=response(output=[text_item("Paris")]))
    )

    assert result.finish_reason is FinishReason.STOP


@pytest.mark.parametrize(
    ("reported", "normalised"),
    [
        ("stop", FinishReason.STOP),
        ("tool_calls", FinishReason.TOOL_CALLS),
        ("function_call", FinishReason.TOOL_CALLS),
        ("length", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("something_new", FinishReason.OTHER),
    ],
)
def test_every_completion_finish_reason_normalises(
    reported: str, normalised: FinishReason
) -> None:
    result = CompletionResult(
        message=ChatCompletionMessage(role="assistant", content="x"),
        finish_reason=reported,
    )

    assert OpenAIService._to_finish_reason(result, None) is normalised


def test_a_completion_reporting_no_reason_falls_back_to_what_it_returned() -> None:
    silent = CompletionResult(
        message=ChatCompletionMessage(role="assistant", content="x"),
        finish_reason=None,
    )

    assert OpenAIService._to_finish_reason(silent, None) is FinishReason.STOP
    assert OpenAIService._to_finish_reason(silent, []) is FinishReason.STOP


def test_completion_usage_folds_into_the_neutral_counts() -> None:
    counts = service()._to_usage_counts(
        completion_usage(
            prompt_tokens_details=PromptTokensDetails(cached_tokens=40),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=7),
        )
    )

    assert counts is not None
    assert counts.input_tokens == 100
    assert counts.output_tokens == 20
    assert counts.cached_input_tokens == 40
    assert counts.reasoning_tokens == 7


def test_response_usage_folds_into_the_neutral_counts() -> None:
    counts = service()._to_usage_counts(
        response_usage(
            input_tokens_details=InputTokensDetails(
                cached_tokens=40, cache_write_tokens=11
            ),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=7),
        )
    )

    assert counts is not None
    assert counts.input_tokens == 100
    assert counts.output_tokens == 20
    assert counts.cached_input_tokens == 40
    assert counts.cache_write_tokens == 11
    assert counts.reasoning_tokens == 7


def test_openai_reports_cached_tokens_as_a_share_of_the_input_total() -> None:
    """Unlike Anthropic, OpenAI's cache counts are already inside the input."""
    counts = service()._to_usage_counts(
        completion_usage(
            prompt_tokens=100,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=90),
        )
    )

    assert counts is not None
    assert counts.input_tokens == 100
    assert counts.cached_input_tokens <= counts.input_tokens


def test_a_call_that_errored_before_usage_arrived_counts_nothing() -> None:
    assert service()._to_usage_counts(None) is None


def test_usage_without_details_still_counts_what_it_has() -> None:
    counts = service()._to_usage_counts(completion_usage())

    assert counts is not None
    assert counts.input_tokens == 100
    assert (counts.cached_input_tokens, counts.reasoning_tokens) == (0, 0)


def test_token_counts_are_read_from_whichever_shape_arrived() -> None:
    assert OpenAIService._extract_token_counts(completion_usage()) == (100, 20, 120)
    assert OpenAIService._extract_token_counts(response_usage()) == (100, 20, 120)
    assert OpenAIService._extract_token_counts(None) == (0, 0, 0)


def test_a_missing_total_is_derived_rather_than_reported_as_zero() -> None:
    assert OpenAIService._extract_token_counts(completion_usage(total_tokens=0)) == (
        100,
        20,
        120,
    )


def test_cache_tokens_are_read_from_whichever_shape_arrived() -> None:
    assert OpenAIService._extract_cache_tokens(
        response_usage(
            input_tokens_details=InputTokensDetails(
                cached_tokens=40, cache_write_tokens=11
            )
        )
    ) == (40, 11)
    assert OpenAIService._extract_cache_tokens(
        completion_usage(prompt_tokens_details=PromptTokensDetails(cached_tokens=40))
    ) == (40, 0)
    assert OpenAIService._extract_cache_tokens(None) == (0, 0)


def test_a_completed_reply_passes_the_incompleteness_check() -> None:
    assert OpenAIService._raise_if_incomplete(response()) is None


def test_a_reply_cut_short_by_the_filter_is_raised_as_a_filter_error() -> None:
    cut = response(
        status="incomplete",
        incomplete_details=IncompleteDetails(reason="content_filter"),
    )

    with pytest.raises(LLMContentFilterError, match="blocked by content filter"):
        OpenAIService._raise_if_incomplete(cut)


def test_a_reply_cut_short_for_any_other_reason_is_raised_as_incomplete() -> None:
    cut = response(
        status="incomplete",
        incomplete_details=IncompleteDetails(reason="max_output_tokens"),
    )

    with pytest.raises(LLMIncompleteError, match="max_output_tokens"):
        OpenAIService._raise_if_incomplete(cut)


def test_an_empty_reply_carries_no_content_rather_than_an_empty_string() -> None:
    result = service()._to_llm_result(ResponseResult(response=response()))

    assert result.content is None
