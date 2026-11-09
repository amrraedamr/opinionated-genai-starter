"""The realtime adapter against a real provider.

Everything else about this adapter can be tested against recorded events. Four
things cannot, and each of them is silent when it breaks:

* whether the gateway proxies a websocket upgrade at all;
* whether the event names this adapter matches are the ones being sent;
* whether real speech survives the trip, which only fails once a sample
  exceeds 0x7f and never does in a hand-written fixture;
* whether answering a tool actually lets the model carry on.
"""

import asyncio
import json
from typing import Any

import pytest

from base.models.definitions.model import FinishReason, ProviderType
from base.models.definitions.realtime_session import (
    AUDIO_SAMPLE_RATE_HZ,
    RealtimeModality,
)
from base.models.view_models.model import ModelConfigurations
from base.models.view_models.realtime_session import (
    RealtimeAudioDelta,
    RealtimeAudioInput,
    RealtimeError,
    RealtimeOutput,
    RealtimeSessionConfig,
    RealtimeTextDelta,
    RealtimeTextInput,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeTurnEnd,
)
from base.services.llm.realtime.openai import OpenAIRealtimeSession
from tests.conftest import credential

MODEL = "gpt-realtime"
TRAILING_SILENCE = bytes(2 * AUDIO_SAMPLE_RATE_HZ)
"""One second of it, at two bytes a sample."""
CONNECT_TIMEOUT = 45.0
TURN_TIMEOUT = 90.0

WEATHER_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def live_session(
    tool_schemas: list[dict[str, Any]] | None = None, **config: Any
) -> OpenAIRealtimeSession:
    return OpenAIRealtimeSession(
        RealtimeSessionConfig(model=MODEL, **config),
        ModelConfigurations(
            api_host=credential("OPENAI_LLM_ENDPOINT"),
            api_key=credential("OPENAI_LLM_API_KEY"),
            model_name=MODEL,
            api_type=ProviderType.OPENAI,
        ),
        tool_schemas=tool_schemas,
    )


async def until_turn_end(
    session: OpenAIRealtimeSession, collect: list[RealtimeOutput]
) -> RealtimeTurnEnd | None:
    """Drain one turn, keeping what arrived. Errors stop it like a turn would."""
    async for event in session.areceive():
        collect.append(event)
        if isinstance(event, RealtimeTurnEnd):
            return event
        if isinstance(event, RealtimeError):
            return None
    return None


@pytest.mark.live
async def test_a_realtime_session_opens_through_the_gateway() -> None:
    """A websocket upgrade is not an HTTP request, and a proxy may refuse it."""
    session = live_session()
    await asyncio.wait_for(session.aconnect(), timeout=CONNECT_TIMEOUT)
    await session.aclose()


@pytest.mark.live
async def test_a_typed_question_is_answered() -> None:
    """Proves the event names this adapter matches are the ones being sent.

    Matching names the provider no longer uses connects happily and yields
    nothing, so an empty reply here is the whole failure.
    """
    session = live_session(instructions="Answer in one short sentence.")
    await asyncio.wait_for(session.aconnect(), timeout=CONNECT_TIMEOUT)
    try:
        await session.asend(RealtimeTextInput(text="Name the capital of France."))
        events: list[RealtimeOutput] = []
        end = await asyncio.wait_for(
            until_turn_end(session, events), timeout=TURN_TIMEOUT
        )

        said = "".join(e.text for e in events if isinstance(e, RealtimeTextDelta))
        assert "Paris" in said
        assert end is not None
        assert end.finish_reason == FinishReason.STOP
    finally:
        await session.aclose()


@pytest.mark.live
async def test_a_turn_reports_what_it_consumed() -> None:
    """Billing cannot wait for a session that stays open for an hour."""
    session = live_session(instructions="Answer in one word.")
    await asyncio.wait_for(session.aconnect(), timeout=CONNECT_TIMEOUT)
    try:
        await session.asend(RealtimeTextInput(text="Say hello."))
        end = await asyncio.wait_for(until_turn_end(session, []), timeout=TURN_TIMEOUT)

        assert end is not None
        assert end.usage is not None
        assert end.usage.input_tokens > 0
        assert end.usage.output_tokens > 0
    finally:
        await session.aclose()


@pytest.mark.live
async def test_real_speech_survives_the_trip() -> None:
    """The failure a fixture cannot reproduce.

    Speech is arbitrary bytes. Plain `bytes` only fails once a sample exceeds
    0x7f, which quiet invented audio never does and a real voice does within
    the first syllable, so this asserts the provider actually sent one.
    """
    session = live_session(
        modalities=[RealtimeModality.AUDIO],
        voice="alloy",
        instructions="Answer in one short sentence.",
    )
    await asyncio.wait_for(session.aconnect(), timeout=CONNECT_TIMEOUT)
    try:
        await session.asend(RealtimeTextInput(text="Say hello and name a colour."))
        events: list[RealtimeOutput] = []
        await asyncio.wait_for(until_turn_end(session, events), timeout=TURN_TIMEOUT)

        spoken = b"".join(e.audio for e in events if isinstance(e, RealtimeAudioDelta))
        assert spoken, "no audio arrived"
        assert max(spoken) > 0x7F, "no sample above 0x7f; the risky path went untested"
    finally:
        await session.aclose()


@pytest.mark.live
async def test_an_audio_session_still_produces_text() -> None:
    """The model's own transcript is the only place its words arrive as text.

    Translating that event as a user transcript would leave an audio session
    with nothing to caption, and attribute the model's sentence to the user.
    """
    session = live_session(
        modalities=[RealtimeModality.AUDIO],
        voice="alloy",
        instructions="Answer in one short sentence.",
    )
    await asyncio.wait_for(session.aconnect(), timeout=CONNECT_TIMEOUT)
    try:
        await session.asend(RealtimeTextInput(text="Name the capital of France."))
        events: list[RealtimeOutput] = []
        await asyncio.wait_for(until_turn_end(session, events), timeout=TURN_TIMEOUT)

        captions = "".join(e.text for e in events if isinstance(e, RealtimeTextDelta))
        assert "Paris" in captions
    finally:
        await session.aclose()


@pytest.mark.live
async def test_answering_a_tool_lets_the_model_carry_on() -> None:
    """It stopped when it asked. Without a response request it waits forever."""
    session = live_session(
        tool_schemas=WEATHER_TOOL,
        instructions="Use the tool for weather questions. Answer in one sentence.",
    )
    await asyncio.wait_for(session.aconnect(), timeout=CONNECT_TIMEOUT)
    try:
        await session.asend(RealtimeTextInput(text="What is the weather in Cairo?"))

        said = ""
        asked = False
        turns = 0

        async def converse() -> None:
            nonlocal said, asked, turns
            async for event in session.areceive():
                if isinstance(event, RealtimeToolCall):
                    asked = True
                    assert event.call.function.name == "get_weather"
                    assert "Cairo" in event.call.function.arguments
                    await session.asend_tool_result(
                        event.call.id, json.dumps({"tempC": 31, "sky": "clear"})
                    )
                elif isinstance(event, RealtimeTextDelta):
                    said += event.text
                elif isinstance(event, RealtimeTurnEnd):
                    turns += 1
                    if turns >= 2 or not asked:
                        return

        await asyncio.wait_for(converse(), timeout=TURN_TIMEOUT)

        assert asked, "the model never asked for the tool"
        assert "31" in said, f"the model ignored the tool result: {said!r}"
    finally:
        await session.aclose()


@pytest.mark.live
async def test_interrupting_stops_the_model_mid_answer() -> None:
    """And is not reported as the model deciding it had finished."""
    session = live_session(instructions="Be very verbose.")
    await asyncio.wait_for(session.aconnect(), timeout=CONNECT_TIMEOUT)
    try:
        await session.asend(
            RealtimeTextInput(text="Count from 1 to 200 with commentary on each.")
        )

        seen = 0
        end: RealtimeTurnEnd | None = None

        async def cut_in() -> None:
            nonlocal seen, end
            async for event in session.areceive():
                if isinstance(event, RealtimeTextDelta):
                    seen += 1
                    if seen == 5:
                        await session.ainterrupt()
                elif isinstance(event, RealtimeTurnEnd):
                    end = event
                    return

        await asyncio.wait_for(cut_in(), timeout=TURN_TIMEOUT)

        assert end is not None, "the model never stopped"
        assert end.finish_reason is not FinishReason.STOP
    finally:
        await session.aclose()


@pytest.mark.live
async def test_speech_is_heard_and_reported_back() -> None:
    """The input half, and the setting that silently disables it.

    Input transcription is off unless a session asks, so without that request
    the user transcript never arrives and the adapter declares an event it can
    never produce. The audio fed in is real speech, generated by the provider
    itself, so no microphone is needed to prove the path.
    """
    session = live_session(instructions="Reply briefly.")
    await asyncio.wait_for(session.aconnect(), timeout=CONNECT_TIMEOUT)
    try:
        spoken = await _speech_of(session, "Hello.")

        heard = ""
        replied = ""

        async def listen() -> None:
            nonlocal heard, replied
            for start in range(0, len(spoken), 4800):
                await session.asend(
                    RealtimeAudioInput(audio=spoken[start : start + 4800])
                )
            async for event in session.areceive():
                if isinstance(event, RealtimeTranscriptDelta):
                    heard += event.text
                elif isinstance(event, RealtimeTextDelta):
                    replied += event.text
                if heard and replied:
                    return

        await asyncio.wait_for(listen(), timeout=TURN_TIMEOUT)

        assert heard, "nothing was transcribed; input transcription is off"
        assert replied, "the model heard nothing to answer"
    finally:
        await session.aclose()


@pytest.mark.live
async def test_speech_ends_a_turn_without_being_told_to() -> None:
    """Why audio does not request a reply per chunk.

    The server decides when someone stopped talking. If it did not, this
    session would wait forever, and the adapter would have to guess instead.
    """
    session = live_session(instructions="Reply briefly.")
    await asyncio.wait_for(session.aconnect(), timeout=CONNECT_TIMEOUT)
    try:
        spoken = await _speech_of(session, "Hello there.")
        for start in range(0, len(spoken), 4800):
            await session.asend(RealtimeAudioInput(audio=spoken[start : start + 4800]))

        end = await asyncio.wait_for(until_turn_end(session, []), timeout=TURN_TIMEOUT)

        assert end is not None, "no turn ended; the server did not detect the end"
    finally:
        await session.aclose()


async def _speech_of(session: OpenAIRealtimeSession, words: str) -> bytes:
    """Real speech to feed back in, so the test needs no microphone.

    Trailed by silence, because the server detects the end of a turn by
    hearing someone stop. Generated speech ends on the last syllable, so
    without it the session waits for a pause that never comes — which a real
    microphone, always sending the room, never fails to provide.
    """
    speaker = live_session(
        modalities=[RealtimeModality.AUDIO],
        voice="alloy",
        instructions=f"Say exactly this and nothing else: {words}",
    )
    await asyncio.wait_for(speaker.aconnect(), timeout=CONNECT_TIMEOUT)
    try:
        await speaker.asend(RealtimeTextInput(text=words))
        events: list[RealtimeOutput] = []
        await asyncio.wait_for(until_turn_end(speaker, events), timeout=TURN_TIMEOUT)
        spoken = b"".join(e.audio for e in events if isinstance(e, RealtimeAudioDelta))
        assert spoken, "the provider generated no speech to feed back"
        return spoken + TRAILING_SILENCE
    finally:
        await speaker.aclose()
