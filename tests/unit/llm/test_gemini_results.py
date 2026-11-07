"""Pins how a Gemini reply becomes the neutral result.

Two counts need normalising, and they need it in opposite directions.
`thoughts_token_count` sits *beside* `candidates_token_count`, so it is added
to the output total; `cached_content_token_count` is already *inside*
`prompt_token_count`, so it is not. Getting either wrong changes what a call
appears to have cost.
"""

from typing import Any

import pytest
from google.genai import types as genai_types

from base.models.definitions.model import FinishReason
from base.models.view_models import BaseModel
from base.models.view_models.exceptions.llm import (
    LLMContentFilterError,
    LLMError,
    LLMStreamError,
)
from base.services.llm.gemini import GeminiService


class Answer(BaseModel):
    """The target of a parse call."""

    text: str


def service() -> GeminiService:
    """A service with only what the result methods read."""
    bare = object.__new__(GeminiService)
    bare._model = "gemini-test"
    return bare


def response(
    *parts: genai_types.Part,
    finish: genai_types.FinishReason | None = genai_types.FinishReason.STOP,
    usage: genai_types.GenerateContentResponseUsageMetadata | None = None,
) -> genai_types.GenerateContentResponse:
    """A reply carrying the parts a test cares about."""
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=list(parts)),
                finish_reason=finish,
            )
        ],
        usage_metadata=usage,
    )


def usage(**counts: int) -> genai_types.GenerateContentResponseUsageMetadata:
    return genai_types.GenerateContentResponseUsageMetadata(**counts)


def test_text_and_thought_parts_are_separated() -> None:
    result = service()._to_result(
        response(
            genai_types.Part(text="weighing it up", thought=True),
            genai_types.Part(text="Paris"),
        )
    )

    assert result.content == "Paris"
    assert result.reasoning == "weighing it up"


def test_a_function_call_part_becomes_a_tool_call() -> None:
    result = service()._to_result(
        response(
            genai_types.Part(
                function_call=genai_types.FunctionCall(
                    name="get_weather", args={"city": "Paris"}
                )
            ),
            finish=genai_types.FinishReason.STOP,
        )
    )

    assert result.tool_calls is not None
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[0].function.arguments == '{"city": "Paris"}'


def test_a_function_call_with_no_arguments_sends_an_empty_object() -> None:
    result = service()._to_result(
        response(
            genai_types.Part(
                function_call=genai_types.FunctionCall(name="ping", args=None)
            )
        )
    )

    assert result.tool_calls is not None
    assert result.tool_calls[0].function.arguments == "{}"


def test_a_reply_with_no_parts_carries_no_content() -> None:
    result = service()._to_result(response())

    assert result.content is None
    assert result.tool_calls is None
    assert result.reasoning is None


@pytest.mark.parametrize(
    ("reported", "normalised"),
    [
        (genai_types.FinishReason.STOP, FinishReason.STOP),
        (genai_types.FinishReason.MAX_TOKENS, FinishReason.LENGTH),
        (genai_types.FinishReason.MALFORMED_FUNCTION_CALL, FinishReason.OTHER),
        (genai_types.FinishReason.OTHER, FinishReason.OTHER),
        (None, FinishReason.OTHER),
    ],
)
def test_every_finish_reason_that_returns_a_reply_normalises(
    reported: genai_types.FinishReason | None, normalised: FinishReason
) -> None:
    result = service()._to_result(response(genai_types.Part(text="x"), finish=reported))

    assert result.finish_reason is normalised


@pytest.mark.parametrize(
    "blocked",
    [
        genai_types.FinishReason.SAFETY,
        genai_types.FinishReason.PROHIBITED_CONTENT,
        genai_types.FinishReason.BLOCKLIST,
        genai_types.FinishReason.SPII,
        genai_types.FinishReason.IMAGE_SAFETY,
        genai_types.FinishReason.RECITATION,
    ],
)
def test_every_blocking_finish_reason_is_raised_rather_than_returned(
    blocked: genai_types.FinishReason,
) -> None:
    with pytest.raises(LLMContentFilterError, match=blocked.name):
        service()._to_result(response(genai_types.Part(text="x"), finish=blocked))


def test_a_reply_with_no_candidates_still_normalises() -> None:
    empty = genai_types.GenerateContentResponse(candidates=[])

    result = service()._to_result(empty)

    assert result.content is None
    assert result.finish_reason is FinishReason.OTHER


def test_thinking_tokens_are_added_to_the_output_total() -> None:
    counts = GeminiService._to_usage_counts(
        response(usage=usage(candidates_token_count=20, thoughts_token_count=7))
    )

    assert counts.output_tokens == 27
    assert counts.reasoning_tokens == 7


def test_cached_tokens_are_already_inside_the_input_total() -> None:
    counts = GeminiService._to_usage_counts(
        response(usage=usage(prompt_token_count=100, cached_content_token_count=40))
    )

    assert counts.input_tokens == 100
    assert counts.cached_input_tokens == 40


def test_gemini_reports_no_cache_writes() -> None:
    counts = GeminiService._to_usage_counts(
        response(usage=usage(prompt_token_count=100, cached_content_token_count=40))
    )

    assert counts.cache_write_tokens == 0


def test_a_reply_with_no_usage_counts_nothing() -> None:
    counts = GeminiService._to_usage_counts(response())

    assert counts.input_tokens == 0
    assert counts.output_tokens == 0


def test_structured_output_is_read_from_the_reply_text() -> None:
    result = service()._to_parsed_result(
        Answer, response(genai_types.Part(text='{"text": "Paris"}'))
    )

    assert result.parsed == Answer(text="Paris")


def test_a_structured_reply_carries_its_usage() -> None:
    result = service()._to_parsed_result(
        Answer,
        response(
            genai_types.Part(text='{"text": "Paris"}'),
            usage=usage(prompt_token_count=5, candidates_token_count=3),
        ),
    )

    assert result.usage is not None
    assert result.usage.total_tokens == 8


def test_an_empty_structured_reply_is_refused() -> None:
    with pytest.raises(LLMError, match="no structured output"):
        service()._to_parsed_result(Answer, response())


def test_structured_output_that_is_not_json_is_refused() -> None:
    with pytest.raises(LLMError, match="not valid JSON"):
        service()._to_parsed_result(
            Answer, response(genai_types.Part(text="Paris, obviously."))
        )


def test_streamed_chunks_are_joined_into_one_result() -> None:
    chunks = [
        response(genai_types.Part(text="Pa")),
        response(
            genai_types.Part(text="ris"),
            usage=usage(prompt_token_count=5, candidates_token_count=3),
        ),
    ]

    result = service()._to_result_from_chunks(chunks)

    assert result.content == "Paris"
    assert result.usage is not None
    assert result.usage.input_tokens == 5


def test_the_last_chunk_carrying_usage_wins() -> None:
    chunks = [
        response(
            genai_types.Part(text="Pa"),
            usage=usage(prompt_token_count=5, candidates_token_count=1),
        ),
        response(
            genai_types.Part(text="ris"),
            usage=usage(prompt_token_count=5, candidates_token_count=3),
        ),
        response(genai_types.Part(text="!")),
    ]

    result = service()._to_result_from_chunks(chunks)

    assert result.usage is not None
    assert result.usage.output_tokens == 3


def test_a_stream_that_produced_nothing_is_refused() -> None:
    with pytest.raises(LLMStreamError, match="no chunks"):
        service()._to_result_from_chunks([])


def test_a_stream_error_currently_drops_the_message_it_was_given() -> None:
    """Product bug: `StreamError` has no `message` field, so the text is lost."""
    error = GeminiService._build_stream_error(RuntimeError("boom"))

    assert error.text is None
    assert error.detail == {
        "exception_type": "RuntimeError",
        "exception_module": "builtins",
    }


def test_the_stream_iterator_has_no_put_method_for_the_streamer_to_call() -> None:
    """Product bug: `GeminiService._stream` calls `stream_iterator.put`."""
    from base.tools.stream import StreamIterator

    assert not hasattr(StreamIterator, "put")
    assert hasattr(StreamIterator, "send")


def test_a_stream_message_cannot_be_built_the_way_the_streamer_builds_it() -> None:
    """Product bug: `GeminiService._stream` passes `content=` where `data=` is."""
    from pydantic import ValidationError

    from base.models.view_models.streams import StreamMessage

    with pytest.raises(ValidationError, match="data"):
        StreamMessage(content="chunk")  # type: ignore[call-arg]


def test_unicode_and_emoji_survive_the_result() -> None:
    result = service()._to_result(response(genai_types.Part(text="héllo 🌍 café")))

    assert result.content == "héllo 🌍 café"


def test_parts_are_read_off_whatever_the_candidate_carries() -> None:
    candidate: Any = genai_types.Candidate(content=None)

    assert GeminiService._parts(candidate) == []
    assert GeminiService._finish_name(candidate) == ""
    assert GeminiService._first_candidate(candidate) is None
