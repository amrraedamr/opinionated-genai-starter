"""Translating Gemini Live messages into the neutral ones.

Built from the SDK's own types. Gemini's shape differs from OpenAI's in ways
the translation has to absorb: text and audio are parts of one turn rather
than separate events, a single server message can carry several of ours, and
the model is told about a barge-in rather than asked to cancel.
"""

import json
import os
from typing import Any

import pytest
from google.genai.types import (
    Blob,
    Content,
    FunctionCall,
    LiveServerContent,
    LiveServerMessage,
    LiveServerToolCall,
    Part,
    Transcription,
    UsageMetadata,
)

from base.models.definitions.model import FinishReason
from base.models.definitions.realtime_session import (
    AUDIO_OUTPUT_SAMPLE_RATE_HZ,
    RealtimeModality,
)
from base.models.view_models.realtime_session import (
    RealtimeAudioDelta,
    RealtimeAudioInput,
    RealtimeSessionConfig,
    RealtimeTextDelta,
    RealtimeTextInput,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeTurnEnd,
)
from base.services.llm.realtime.gemini import (
    INPUT_SAMPLE_RATE_HZ,
    GeminiLiveSession,
)


class FakeLive:
    """Records what was sent and replays a script of server messages."""

    def __init__(self, script: list[LiveServerMessage] | None = None) -> None:
        self.realtime: list[dict[str, Any]] = []
        self.client_content: list[dict[str, Any]] = []
        self.tool_responses: list[Any] = []
        self.script = script or []

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime.append(kwargs)

    async def send_client_content(self, **kwargs: Any) -> None:
        self.client_content.append(kwargs)

    async def send_tool_response(self, **kwargs: Any) -> None:
        self.tool_responses.append(kwargs)

    async def receive(self) -> Any:
        for message in self.script:
            yield message


def session(
    script: list[LiveServerMessage] | None = None, **config: Any
) -> GeminiLiveSession:
    live = GeminiLiveSession(RealtimeSessionConfig(model="gemini-live", **config))
    live._live = FakeLive(script)  # type: ignore[assignment]
    return live


def fake(live: GeminiLiveSession) -> FakeLive:
    return live._live  # type: ignore[return-value]


async def translate(script: list[LiveServerMessage]) -> list[Any]:
    return [event async for event in session(script).areceive()]


def spoken(text: str = "", audio: bytes = b"", **content: Any) -> LiveServerMessage:
    parts: list[Part] = []
    if text:
        parts.append(Part(text=text))
    if audio:
        parts.append(Part(inline_data=Blob(data=audio, mime_type="audio/pcm")))
    return LiveServerMessage(
        server_content=LiveServerContent(
            model_turn=Content(role="model", parts=parts) if parts else None, **content
        )
    )


# Rates, which differ and fail silently


def test_gemini_listens_at_a_different_rate_than_openai() -> None:
    """Sending the wrong one raises nothing; the model hears a chipmunk."""
    assert INPUT_SAMPLE_RATE_HZ == 16_000
    assert session().input_sample_rate == 16_000


def test_both_providers_still_speak_at_the_same_rate() -> None:
    assert AUDIO_OUTPUT_SAMPLE_RATE_HZ == 24_000


def test_outgoing_audio_states_the_rate_it_was_sampled_at() -> None:
    """The provider cannot infer it, and guesses wrong in silence."""
    live = session()
    import asyncio

    asyncio.run(_send_audio(live))
    blob = fake(live).realtime[0]["audio"]
    assert f"rate={INPUT_SAMPLE_RATE_HZ}" in blob.mime_type


async def _send_audio(live: GeminiLiveSession) -> None:
    await live.asend(RealtimeAudioInput(audio=b"\x01\x02"))


# Translation


async def test_text_in_a_turn_is_translated() -> None:
    [out] = await translate([spoken(text="hello")])
    assert isinstance(out, RealtimeTextDelta)
    assert out.text == "hello"


async def test_audio_in_a_turn_is_translated_as_samples() -> None:
    """Gemini hands back bytes already; nothing should re-encode them."""
    pcm = os.urandom(64)
    [out] = await translate([spoken(audio=pcm)])
    assert isinstance(out, RealtimeAudioDelta)
    assert out.audio == pcm


async def test_one_message_carrying_two_things_yields_both() -> None:
    """A turn can speak and finish in the same breath.

    Reducing a message to whichever event was noticed first would drop the
    other, and the one usually dropped is the end of the turn.
    """
    events = await translate([spoken(text="bye", turn_complete=True)])
    assert [type(e) for e in events] == [RealtimeTextDelta, RealtimeTurnEnd]


async def test_the_users_words_are_attributed_to_the_user() -> None:
    [out] = await translate(
        [
            LiveServerMessage(
                server_content=LiveServerContent(
                    input_transcription=Transcription(text="what I said")
                )
            )
        ]
    )
    assert isinstance(out, RealtimeTranscriptDelta)


async def test_the_models_own_transcript_is_its_text() -> None:
    """In an audio session it is the only place the model's words arrive."""
    [out] = await translate(
        [
            LiveServerMessage(
                server_content=LiveServerContent(
                    output_transcription=Transcription(text="what it said")
                )
            )
        ]
    )
    assert isinstance(out, RealtimeTextDelta)


async def test_a_tool_call_carries_json_arguments() -> None:
    """Gemini parses them; the neutral shape carries the string form."""
    [out] = await translate(
        [
            LiveServerMessage(
                tool_call=LiveServerToolCall(
                    function_calls=[
                        FunctionCall(id="c1", name="lookup", args={"q": "x"})
                    ]
                )
            )
        ]
    )
    assert isinstance(out, RealtimeToolCall)
    assert json.loads(out.call.function.arguments) == {"q": "x"}


async def test_a_completed_turn_reports_a_clean_stop() -> None:
    [out] = await translate([spoken(turn_complete=True)])
    assert isinstance(out, RealtimeTurnEnd)
    assert out.finish_reason == FinishReason.STOP


async def test_an_interrupted_turn_is_not_reported_as_a_clean_stop() -> None:
    [out] = await translate([spoken(turn_complete=True, interrupted=True)])
    assert out.finish_reason == FinishReason.OTHER  # type: ignore[union-attr]


async def test_usage_is_reported_when_the_turn_ends() -> None:
    message = spoken(turn_complete=True)
    message.usage_metadata = UsageMetadata(
        prompt_token_count=12, response_token_count=7
    )
    [out] = await translate([message])
    assert out.usage is not None  # type: ignore[union-attr]
    assert out.usage.input_tokens == 12  # type: ignore[union-attr]


async def test_a_message_with_nothing_in_it_yields_nothing() -> None:
    assert await translate([LiveServerMessage()]) == []


# Sending


async def test_typed_text_is_sent_as_a_finished_turn() -> None:
    live = session()
    await live.asend(RealtimeTextInput(text="hello"))
    assert fake(live).client_content[0]["turn_complete"] is True


async def test_audio_is_not_sent_as_a_finished_turn() -> None:
    """The server detects the end. Marking each chunk complete would start a
    turn for every packet of microphone input."""
    live = session()
    await live.asend(RealtimeAudioInput(audio=b"\x01"))
    assert fake(live).client_content == []
    assert fake(live).realtime


async def test_answering_a_tool_names_the_function_not_the_call() -> None:
    """Gemini needs the name; the seam only carries the id.

    Remembering which function a call belonged to keeps that difference out
    of the seam, which would otherwise carry a field only Gemini reads.
    """
    live = session()
    live._tool_names["c1"] = "lookup"

    await live.asend_tool_result("c1", '{"ok":true}')

    [response] = fake(live).tool_responses[0]["function_responses"]
    assert response.name == "lookup"
    assert response.id == "c1"


async def test_a_remembered_call_is_only_answered_once() -> None:
    live = session()
    live._tool_names["c1"] = "lookup"
    await live.asend_tool_result("c1", "{}")
    assert "c1" not in live._tool_names


async def test_a_tool_call_is_remembered_when_it_arrives() -> None:
    live = session(
        [
            LiveServerMessage(
                tool_call=LiveServerToolCall(
                    function_calls=[FunctionCall(id="c9", name="lookup", args={})]
                )
            )
        ]
    )
    [_] = [event async for event in live.areceive()]
    assert live._tool_names["c9"] == "lookup"


async def test_interrupting_announces_activity_rather_than_cancelling() -> None:
    """There is no cancel message. Gemini stops when fresh input arrives."""
    live = session()
    await live.ainterrupt()
    assert "activity_start" in fake(live).realtime[0]


# Configuration


def test_both_transcriptions_are_asked_for() -> None:
    """Off by default. Without them a caller can neither show what was heard
    nor caption an audio reply."""
    config = session()._connect_config()
    assert config.input_audio_transcription is not None
    assert config.output_audio_transcription is not None


def test_modalities_are_spelled_the_way_gemini_wants() -> None:
    config = session(modalities=[RealtimeModality.AUDIO])._connect_config()
    assert config.response_modalities == ["AUDIO"]


def test_a_voice_without_audio_output_is_not_sent() -> None:
    assert session(voice="Puck")._connect_config().speech_config is None


def test_a_voice_is_sent_when_the_session_speaks() -> None:
    config = session(
        voice="Puck", modalities=[RealtimeModality.AUDIO]
    )._connect_config()
    assert config.speech_config is not None


def test_instructions_are_omitted_rather_than_sent_empty() -> None:
    assert session()._connect_config().system_instruction is None


# Lifecycle


async def test_using_a_session_before_connecting_says_so() -> None:
    live = GeminiLiveSession(RealtimeSessionConfig(model="m"))
    with pytest.raises(RuntimeError, match="before aconnect"):
        await live.asend(RealtimeTextInput(text="hi"))


async def test_closing_a_session_that_never_opened_is_quiet() -> None:
    await GeminiLiveSession(RealtimeSessionConfig(model="m")).aclose()


async def test_a_socket_that_fails_to_close_does_not_raise() -> None:
    class Angry:
        async def __aexit__(self, *_: object) -> None:
            raise OSError("already gone")

    live = GeminiLiveSession(RealtimeSessionConfig(model="m"))
    live._manager = Angry()
    await live.aclose()


def test_the_service_names_itself() -> None:
    assert session().get_service_name() == "gemini_live"
