"""What a model call returns: `LLMResult`, `LLMUsageCounts`, `ParsedResult`,
`ToolRunResult`.

These are the shapes every provider adapter normalises into, so a caller can
branch on `finish_reason` without knowing which provider answered.
"""

from uuid import UUID

import pytest
from pydantic import ValidationError

from base.models.definitions.message import MessageRole
from base.models.definitions.model import FinishReason
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredMessage, StoredToolCall
from base.models.view_models.llm_result import (
    LLMResult,
    LLMUsageCounts,
    ParsedResult,
    ToolRunResult,
)
from base.models.view_models.usage import ProviderUsage


def tool_call(identifier: str = "call_1") -> StoredToolCall:
    return StoredToolCall(
        id=identifier,
        function={"name": "get_weather", "arguments": '{"city":"Cairo"}'},  # type: ignore[arg-type]
    )


def test_an_llm_result_defaults_to_an_empty_answer() -> None:
    result = LLMResult()
    assert result.content is None
    assert result.tool_calls is None
    assert result.usage is None
    assert result.reasoning is None
    assert result.finish_reason is None


def test_an_llm_result_carries_the_normalised_stop_reason() -> None:
    result = LLMResult(content="hi", finish_reason=FinishReason.LENGTH)
    assert result.finish_reason is FinishReason.LENGTH
    assert result.finish_reason == "length"


def test_an_llm_result_accepts_a_stop_reason_as_its_wire_value() -> None:
    assert (
        LLMResult(finish_reason="tool_calls").finish_reason is FinishReason.TOOL_CALLS
    )


def test_an_unknown_stop_reason_is_rejected_rather_than_passed_through() -> None:
    with pytest.raises(ValidationError):
        LLMResult(finish_reason="exploded")


def test_an_llm_result_carries_the_tool_calls_the_model_asked_for() -> None:
    result = LLMResult(tool_calls=[tool_call()], finish_reason=FinishReason.TOOL_CALLS)
    assert result.tool_calls is not None
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].type == "function"
    assert result.tool_calls[0].function.name == "get_weather"


def test_usage_counts_extend_provider_usage_with_reasoning_tokens() -> None:
    counts = LLMUsageCounts(input_tokens=10, output_tokens=4, reasoning_tokens=3)
    assert isinstance(counts, ProviderUsage)
    assert counts.total_tokens == 14
    assert counts.reasoning_tokens == 3


def test_usage_counts_default_reasoning_tokens_to_zero() -> None:
    assert LLMUsageCounts().reasoning_tokens == 0


def test_an_llm_result_camelises_its_nested_usage_on_dump() -> None:
    result = LLMResult(
        content="hi",
        usage=LLMUsageCounts(input_tokens=1, cached_input_tokens=1, reasoning_tokens=2),
        finish_reason=FinishReason.STOP,
    )
    dumped = result.serialize()
    assert dumped["finishReason"] == "stop"
    assert dumped["usage"]["cachedInputTokens"] == 1
    assert dumped["usage"]["reasoningTokens"] == 2


class Answer(BaseModel):
    city: str
    confidence: float = 1.0


def test_a_parsed_result_holds_the_schema_the_caller_asked_for() -> None:
    parsed: ParsedResult[Answer] = ParsedResult(parsed=Answer(city="Cairo"))
    assert parsed.parsed.city == "Cairo"
    assert parsed.usage is None


def test_a_parsed_result_validates_its_payload_against_that_schema() -> None:
    with pytest.raises(ValidationError):
        ParsedResult[Answer](parsed={"confidence": 0.5})  # type: ignore[arg-type]


def test_a_tool_run_result_reports_the_final_turn_and_how_many_rounds_it_took() -> None:
    run = ToolRunResult(
        result=LLMResult(content="21 degrees", finish_reason=FinishReason.STOP),
        messages=[
            StoredMessage(
                conversation_id=UUID(int=1),
                sequence=1,
                role=MessageRole.ASSISTANT,
                tool_calls=[tool_call()],
            ),
            StoredMessage(
                conversation_id=UUID(int=1),
                sequence=2,
                role=MessageRole.TOOL,
                content="21",
                tool_call_id="call_1",
            ),
        ],
        rounds=2,
    )
    assert run.result.content == "21 degrees"
    assert run.rounds == 2
    assert [m.role for m in run.messages] == [MessageRole.ASSISTANT, MessageRole.TOOL]


def test_a_tool_run_result_requires_every_field_it_reports() -> None:
    with pytest.raises(ValidationError):
        ToolRunResult(result=LLMResult(), messages=[])  # type: ignore[call-arg]
