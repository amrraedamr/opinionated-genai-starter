"""Translating OpenAI's realtime events into the neutral ones.

Built from the SDK's own event classes rather than hand-made stand-ins. The
failure this guards against is a rename: `response.text.delta` became
`response.output_text.delta`, and an adapter matching the old name connects
happily and yields nothing at all. A test with invented objects would agree
with the adapter and both would be wrong together.
"""

import base64
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from openai.types.realtime import (
    ConversationItemInputAudioTranscriptionDeltaEvent,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
)

from base.models.definitions.model import FinishReason
from base.models.definitions.realtime_session import RealtimeModality
from base.models.view_models.realtime_session import (
    RealtimeAudioDelta,
    RealtimeAudioInput,
    RealtimeError,
    RealtimeSessionConfig,
    RealtimeTextDelta,
    RealtimeTextInput,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeTurnEnd,
)
from base.services.llm.realtime.openai import OpenAIRealtimeSession


class FakeConnection:
    """Records what was sent and replays a script of server events."""

    def __init__(self, script: list[Any] | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self.script = script or []
        self.exited = 0

    async def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self.script:
            yield event

    def types(self) -> list[str]:
        return [event["type"] for event in self.sent]


def session(script: list[Any] | None = None, **config: Any) -> OpenAIRealtimeSession:
    """A session already connected to a fake, skipping the real socket."""
    live = OpenAIRealtimeSession(RealtimeSessionConfig(model="gpt-realtime", **config))
    live._connection = FakeConnection(script)  # type: ignore[assignment]
    return live


def fake(live: OpenAIRealtimeSession) -> FakeConnection:
    return live._connection  # type: ignore[return-value]


async def translate(script: list[Any]) -> list[Any]:
    return [event async for event in session(script).areceive()]


# Event names, the thing that silently breaks


async def test_a_text_delta_is_translated() -> None:
    [out] = await translate(
        [
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                delta="hel",
                content_index=0,
                event_id="e",
                item_id="i",
                output_index=0,
                response_id="r",
            )
        ]
    )
    assert isinstance(out, RealtimeTextDelta)
    assert out.text == "hel"


async def test_audio_is_decoded_rather_than_passed_through_as_base64() -> None:
    """The provider sends base64 text; the seam promises samples."""
    pcm = os.urandom(64)

    [out] = await translate(
        [
            ResponseAudioDeltaEvent(
                type="response.output_audio.delta",
                delta=base64.b64encode(pcm).decode(),
                content_index=0,
                event_id="e",
                item_id="i",
                output_index=0,
                response_id="r",
            )
        ]
    )

    assert isinstance(out, RealtimeAudioDelta)
    assert out.audio == pcm


async def test_the_users_words_are_attributed_to_the_user() -> None:
    """One of two transcript events, and they describe different people."""
    [out] = await translate(
        [
            ConversationItemInputAudioTranscriptionDeltaEvent(
                type="conversation.item.input_audio_transcription.delta",
                delta="what the user said",
                event_id="e",
                item_id="i",
            )
        ]
    )
    assert isinstance(out, RealtimeTranscriptDelta)


async def test_the_models_own_words_are_not_a_user_transcript() -> None:
    """In an audio session this is the only place the model's text arrives.

    Translating it as a transcript would put the model's sentence in the
    user's mouth, and leave an audio-only session with no assistant text.
    """
    [out] = await translate(
        [
            ResponseAudioTranscriptDeltaEvent(
                type="response.output_audio_transcript.delta",
                delta="what the model said",
                content_index=0,
                event_id="e",
                item_id="i",
                output_index=0,
                response_id="r",
            )
        ]
    )
    assert isinstance(out, RealtimeTextDelta)


async def test_a_tool_call_carries_its_call_id() -> None:
    [out] = await translate(
        [
            ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                arguments='{"q":"x"}',
                call_id="call-7",
                event_id="e",
                item_id="i",
                name="lookup",
                output_index=0,
                response_id="r",
            )
        ]
    )
    assert isinstance(out, RealtimeToolCall)
    assert out.call.id == "call-7"
    assert out.call.function.name == "lookup"


async def test_an_error_is_surfaced_without_ending_the_session() -> None:
    [out] = await translate(
        [
            RealtimeErrorEvent(
                type="error",
                event_id="e",
                error={"type": "invalid_request_error", "message": "no"},
            )
        ]
    )
    assert isinstance(out, RealtimeError)
    assert out.recoverable


async def test_bookkeeping_events_are_dropped_rather_than_surfaced() -> None:
    """The provider emits a great deal of it; every caller would filter it."""

    class Noise:
        type = "conversation.item.created"

    assert await translate([Noise()]) == []


# Turns


async def test_a_completed_turn_reports_a_normalised_reason() -> None:
    [out] = await translate([_done(status="completed")])
    assert isinstance(out, RealtimeTurnEnd)
    assert out.finish_reason == FinishReason.STOP


async def test_a_truncated_turn_is_reported_as_length() -> None:
    [out] = await translate([_done(status="incomplete")])
    assert out.finish_reason == FinishReason.LENGTH  # type: ignore[union-attr]


async def test_a_cancelled_turn_is_not_pretended_to_be_a_clean_stop() -> None:
    """Interrupting the model is not the model deciding it was finished."""
    [out] = await translate([_done(status="cancelled")])
    assert out.finish_reason == FinishReason.OTHER  # type: ignore[union-attr]


async def test_usage_is_reported_per_turn() -> None:
    """A session has no total until it closes, and billing cannot wait."""
    [out] = await translate(
        [_done(status="completed", usage={"input_tokens": 11, "output_tokens": 5})]
    )
    assert out.usage is not None  # type: ignore[union-attr]
    assert out.usage.input_tokens == 11  # type: ignore[union-attr]


async def test_a_turn_without_usage_does_not_invent_zeroes() -> None:
    [out] = await translate([_done(status="completed")])
    assert out.usage is None  # type: ignore[union-attr]


def _done(status: str, usage: dict[str, int] | None = None) -> ResponseDoneEvent:
    response: dict[str, Any] = {
        "id": "r",
        "object": "realtime.response",
        "status": status,
    }
    if usage:
        response["usage"] = usage
    return ResponseDoneEvent.model_validate(
        {"type": "response.done", "event_id": "e", "response": response}
    )


# Sending


async def test_typed_text_asks_for_a_reply() -> None:
    live = session()
    await live.asend(RealtimeTextInput(text="hello"))
    assert fake(live).types() == ["conversation.item.create", "response.create"]


async def test_audio_does_not_ask_for_a_reply_per_chunk() -> None:
    """The server decides when the user stopped talking.

    Requesting a response on every packet would start a turn per 20ms of
    microphone input.
    """
    live = session()
    await live.asend(RealtimeAudioInput(audio=b"\x01\x02"))
    assert fake(live).types() == ["input_audio_buffer.append"]


async def test_audio_is_base64_on_the_way_out() -> None:
    live = session()
    pcm = os.urandom(32)
    await live.asend(RealtimeAudioInput(audio=pcm))
    assert fake(live).sent[0]["audio"] == base64.b64encode(pcm).decode()


async def test_a_tool_result_lets_the_model_carry_on() -> None:
    """It stopped when it asked. Without the second message it waits forever."""
    live = session()
    await live.asend_tool_result("call-7", '{"ok":true}')
    assert fake(live).types() == ["conversation.item.create", "response.create"]
    assert fake(live).sent[0]["item"]["call_id"] == "call-7"


async def test_interrupting_cancels_the_turn() -> None:
    live = session()
    await live.ainterrupt()
    assert fake(live).types() == ["response.cancel"]


# Configuration and lifecycle


def test_a_session_is_text_only_unless_audio_was_asked_for() -> None:
    live = session()
    assert live._session_payload()["output_modalities"] == ["text"]


def test_a_voice_without_audio_output_is_not_sent() -> None:
    """Meaningless to the provider, and a plausible way to be refused."""
    assert "output" not in session(voice="alloy")._session_payload()["audio"]


def test_a_voice_is_sent_when_the_session_speaks() -> None:
    live = session(voice="alloy", modalities=[RealtimeModality.AUDIO])
    assert live._session_payload()["audio"]["output"]["voice"] == "alloy"


def test_every_session_asks_to_transcribe_what_it_hears() -> None:
    """Off by default, and without it the user transcript never arrives.

    Asked for even when the reply is text, because that describes what comes
    in rather than what goes out, and speaking to a session that answers in
    writing is the ordinary case.
    """
    payload = session()._session_payload()
    assert payload["audio"]["input"]["transcription"]["model"]


def test_instructions_are_omitted_rather_than_sent_empty() -> None:
    assert "instructions" not in session()._session_payload()


async def test_using_a_session_before_connecting_says_so() -> None:
    """Rather than an attribute error from inside the SDK."""
    live = OpenAIRealtimeSession(RealtimeSessionConfig(model="m"))
    with pytest.raises(RuntimeError, match="before aconnect"):
        await live.asend(RealtimeTextInput(text="hi"))


async def test_closing_a_session_that_never_opened_is_quiet() -> None:
    await OpenAIRealtimeSession(RealtimeSessionConfig(model="m")).aclose()


async def test_closing_twice_is_quiet() -> None:
    """A drain and a client disconnect both close; neither should raise."""

    class Manager:
        async def __aexit__(self, *_: object) -> None:
            return None

    live = OpenAIRealtimeSession(RealtimeSessionConfig(model="m"))
    live._manager = Manager()
    await live.aclose()
    await live.aclose()


async def test_a_socket_that_fails_to_close_does_not_raise() -> None:
    """Closing is cleanup. Raising here loses whatever caused the close."""

    class Angry:
        async def __aexit__(self, *_: object) -> None:
            raise OSError("already gone")

    live = OpenAIRealtimeSession(RealtimeSessionConfig(model="m"))
    live._manager = Angry()
    await live.aclose()


def test_the_service_names_itself() -> None:
    assert session().get_service_name() == "openai_realtime"


def test_the_session_reports_its_model() -> None:
    assert session().model == "gpt-realtime"
