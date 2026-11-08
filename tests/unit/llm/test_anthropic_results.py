"""Pins how an Anthropic reply becomes the neutral result.

A schema-constrained reply arrives as ordinary text rather than a tool call,
and may be preceded by a thinking block the reader has to step over. Usage is
the other asymmetry worth naming: Anthropic reports cache reads and writes
*alongside* `input_tokens`, so the neutral shape has to fold them in.
"""

import json
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage
from pydantic import ValidationError

from base.models.definitions.model import FinishReason
from base.models.view_models import BaseModel
from base.models.view_models.exceptions.llm import LLMContentFilterError, LLMError
from base.services.llm.anthropic import AnthropicService


class Answer(BaseModel):
    """The target of a parse call."""

    text: str
    count: int = 0


def service() -> AnthropicService:
    """A service with only what the result methods read."""
    bare = object.__new__(AnthropicService)
    bare._model = "claude-test"
    return bare


def message(
    *blocks: Any, stop_reason: str | None = "end_turn", **usage: int
) -> Message:
    """A reply carrying the blocks a test cares about."""
    return Message(
        id="msg_1",
        content=list(blocks),
        model="claude-test",
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        type="message",
        usage=Usage(**{"input_tokens": 0, "output_tokens": 0, **usage}),
    )


def text(value: str) -> TextBlock:
    return TextBlock(type="text", text=value)


def thinking(value: str = "weighing it up") -> ThinkingBlock:
    return ThinkingBlock(type="thinking", thinking=value, signature="sig")


def test_a_text_block_is_read_as_the_structured_output() -> None:
    result = service()._to_parsed_result(
        Answer, message(text('{"text": "Paris", "count": 2}'))
    )

    assert result.parsed == Answer(text="Paris", count=2)


def test_a_thinking_block_is_stepped_over() -> None:
    result = service()._to_parsed_result(
        Answer, message(thinking(), text('{"text": "Paris"}'))
    )

    assert result.parsed.text == "Paris"


def test_a_reply_with_no_text_block_is_refused() -> None:
    with pytest.raises(LLMError, match="no structured output"):
        service()._to_parsed_result(Answer, message(thinking()))


def test_a_reply_with_nothing_in_it_at_all_is_refused() -> None:
    with pytest.raises(LLMError, match="no structured output"):
        service()._to_parsed_result(Answer, message())


def test_a_blank_text_block_counts_as_no_output() -> None:
    with pytest.raises(LLMError, match="no structured output"):
        service()._to_parsed_result(Answer, message(text("   \n ")))


def test_output_that_is_not_json_raises_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        service()._to_parsed_result(Answer, message(text("Paris, obviously.")))


def test_json_that_does_not_fit_the_schema_raises_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="count"):
        service()._to_parsed_result(
            Answer, message(text('{"text": "Paris", "count": "two"}'))
        )


def test_a_parsed_result_carries_the_usage_of_the_call_that_made_it() -> None:
    result = service()._to_parsed_result(
        Answer,
        message(text('{"text": "Paris"}'), input_tokens=100, output_tokens=20),
    )

    assert result.usage is not None
    assert (result.usage.input_tokens, result.usage.output_tokens) == (100, 20)


def test_text_and_thinking_are_separated_on_a_completion() -> None:
    result = service()._to_result(message(thinking("hmm"), text("Paris")))

    assert result.content == "Paris"
    assert result.reasoning == "hmm"


def test_several_text_blocks_are_joined_without_a_separator() -> None:
    result = service()._to_result(message(text("Pa"), text("ris")))

    assert result.content == "Paris"


def test_a_reply_with_only_a_tool_call_carries_no_content() -> None:
    result = service()._to_result(
        message(
            ToolUseBlock(
                type="tool_use",
                id="toolu_1",
                name="get_weather",
                input={"city": "Paris"},
            ),
            stop_reason="tool_use",
        )
    )

    assert result.content is None
    assert result.tool_calls is not None
    assert result.tool_calls[0].id == "toolu_1"
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert result.finish_reason is FinishReason.TOOL_CALLS


def test_a_refusal_is_raised_rather_than_returned() -> None:
    with pytest.raises(LLMContentFilterError, match="refused"):
        service()._to_result(message(text("no"), stop_reason="refusal"))


@pytest.mark.parametrize(
    ("reported", "normalised"),
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("tool_use", FinishReason.TOOL_CALLS),
        ("max_tokens", FinishReason.LENGTH),
        ("pause_turn", FinishReason.OTHER),
        (None, FinishReason.OTHER),
    ],
)
def test_every_stop_reason_normalises(
    reported: str | None, normalised: FinishReason
) -> None:
    result = service()._to_result(message(text("x"), stop_reason=reported))

    assert result.finish_reason is normalised


def test_cache_reads_and_writes_are_added_to_the_input_total() -> None:
    counts = AnthropicService._to_usage_counts(
        message(
            text("x"),
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=50,
        )
    )

    assert counts.input_tokens == 180
    assert counts.cached_input_tokens == 30
    assert counts.cache_write_tokens == 50


def test_the_folded_cache_counts_stay_a_share_of_the_input_total() -> None:
    counts = AnthropicService._to_usage_counts(
        message(
            text("x"),
            input_tokens=0,
            cache_read_input_tokens=90,
            cache_creation_input_tokens=10,
        )
    )

    assert counts.cached_input_tokens + counts.cache_write_tokens == counts.input_tokens


def test_usage_without_cache_counts_reports_the_input_unchanged() -> None:
    counts = AnthropicService._to_usage_counts(
        message(text("x"), input_tokens=100, output_tokens=20)
    )

    assert counts.input_tokens == 100
    assert (counts.cached_input_tokens, counts.cache_write_tokens) == (0, 0)


def test_a_result_carries_its_usage() -> None:
    result = service()._to_result(
        message(text("Paris"), input_tokens=7, output_tokens=3)
    )

    assert result.usage is not None
    assert result.usage.total_tokens == 10


def test_unicode_and_emoji_survive_the_result() -> None:
    result = service()._to_result(message(text("héllo 🌍 café")))

    assert result.content == "héllo 🌍 café"


# A gateway may accept `output_config.format` without enforcing it, and the
# model then answers with the JSON fenced the way it would write it in prose.
# The schema was asked for either way, so the answer is read either way.


@pytest.mark.parametrize(
    "body",
    [
        '```json\n{"text": "Paris"}\n```',
        '```\n{"text": "Paris"}\n```',
        '```JSON\r\n{"text": "Paris"}\r\n```',
        '  ```json\n{"text": "Paris"}\n```  ',
        '{"text": "Paris"}',
    ],
)
def test_a_schema_is_read_whether_or_not_the_reply_is_fenced(body: str) -> None:
    parsed = AnthropicService._to_parsed_result(
        service(), Answer, message(text(body))
    )

    assert parsed.parsed.text == "Paris"


def test_prose_around_the_fence_does_not_defeat_it() -> None:
    """Some models introduce the block before writing it."""
    body = 'Here is the answer:\n```json\n{"text": "Paris"}\n```'

    parsed = AnthropicService._to_parsed_result(service(), Answer, message(text(body)))

    assert parsed.parsed.text == "Paris"


def test_a_reply_that_is_not_json_at_all_still_fails() -> None:
    """Tolerating a fence must not turn a wrong answer into a silent one."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnthropicService._to_parsed_result(
            service(), Answer, message(text("I cannot."))
        )
