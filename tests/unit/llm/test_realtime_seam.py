"""The shape a realtime provider has to fit, and what it carries.

No adapter exists yet, so these pin the seam itself: that the vocabulary is
provider-neutral, that an event says what it is rather than leaving a caller
to guess from which field is populated, and that audio survives the trip to a
client, which is the one that is easy to get wrong and silent when it is.
"""

import base64
import os
from collections.abc import AsyncIterator

import pytest
from pydantic import TypeAdapter, ValidationError

from base.models.definitions.model import FinishReason
from base.models.definitions.realtime_session import (
    RealtimeEventType,
    RealtimeModality,
)
from base.models.view_models.chat_history import StoredToolCall, StoredToolCallFunction
from base.models.view_models.realtime_session import (
    RealtimeAudioDelta,
    RealtimeAudioInput,
    RealtimeError,
    RealtimeInput,
    RealtimeOutput,
    RealtimeSessionConfig,
    RealtimeTextDelta,
    RealtimeTextInput,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeTurnEnd,
)
from base.services.llm.realtime import AbstractRealtimeSession

OUTPUT = TypeAdapter(RealtimeOutput)
INPUT = TypeAdapter(RealtimeInput)


# Audio


def test_audio_survives_the_trip_to_a_client() -> None:
    """The one that is silent when it breaks.

    PCM is arbitrary bytes, a websocket carries text, and plain `bytes` only
    fails once a sample happens to exceed 0x7f — which a quiet test fixture
    never does and a real microphone does immediately.
    """
    pcm = os.urandom(512)

    restored = RealtimeAudioDelta.model_validate_json(
        RealtimeAudioDelta(audio=pcm).model_dump_json()
    )

    assert restored.audio == pcm


def test_audio_is_base64_on_the_wire_not_mangled_text() -> None:
    pcm = bytes(range(256))
    sent = RealtimeAudioDelta(audio=pcm).model_dump_json()
    assert base64.b64encode(pcm).decode() in sent


def test_audio_stays_bytes_in_python() -> None:
    """Base64 is a wire concern. A caller holding samples should hold samples."""
    delta = RealtimeAudioDelta(audio=b"\x00\x01")
    assert isinstance(delta.model_dump()["audio"], bytes)


def test_audio_input_survives_the_same_trip() -> None:
    pcm = os.urandom(256)
    restored = RealtimeAudioInput.model_validate_json(
        RealtimeAudioInput(audio=pcm).model_dump_json()
    )
    assert restored.audio == pcm


# Events say what they are


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (RealtimeTextDelta(text="hi"), RealtimeEventType.TEXT_DELTA),
        (RealtimeAudioDelta(audio=b"\x01"), RealtimeEventType.AUDIO_DELTA),
        (RealtimeTranscriptDelta(text="hi"), RealtimeEventType.TRANSCRIPT_DELTA),
        (RealtimeTurnEnd(), RealtimeEventType.TURN_END),
        (RealtimeError(message="no"), RealtimeEventType.ERROR),
    ],
)
def test_every_event_names_itself(event: object, expected: RealtimeEventType) -> None:
    assert event.type == expected  # type: ignore[attr-defined]


def test_an_event_is_restored_as_the_member_it_was() -> None:
    """A union rather than one model with everything optional.

    Round-tripping has to give back the same class, or a caller matching on
    type is matching on something the parser invented.
    """
    restored = OUTPUT.validate_json(OUTPUT.dump_json(RealtimeTextDelta(text="hi")))
    assert isinstance(restored, RealtimeTextDelta)


def test_a_tool_call_is_restored_as_a_tool_call() -> None:
    call = RealtimeToolCall(
        call=StoredToolCall(
            id="call-1",
            function=StoredToolCallFunction(name="lookup", arguments="{}"),
        )
    )
    restored = OUTPUT.validate_json(OUTPUT.dump_json(call))
    assert isinstance(restored, RealtimeToolCall)
    assert restored.call.function.name == "lookup"


def test_an_unknown_event_type_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValidationError):
        OUTPUT.validate_python({"type": "something_new", "text": "hi"})


def test_a_text_delta_has_no_audio_to_read() -> None:
    """What the union buys over one model with every field optional.

    Not that extra input is refused — the shared `BaseModel` ignores extras
    everywhere — but that the object a caller ends up holding has exactly the
    fields its event has, so reaching for audio on a text delta is a mistake
    a type checker catches rather than a `None` to handle at runtime.
    """
    assert "audio" not in RealtimeTextDelta.model_fields
    assert "audio" in RealtimeAudioDelta.model_fields


def test_input_is_restored_as_the_member_it_was() -> None:
    restored = INPUT.validate_json(INPUT.dump_json(RealtimeTextInput(text="hi")))
    assert isinstance(restored, RealtimeTextInput)


def test_an_empty_delta_is_not_the_end_of_a_turn() -> None:
    """Why a caller branches on type rather than on an empty field."""
    assert RealtimeTextDelta(text="").type != RealtimeTurnEnd().type


# A turn ends on its own terms


def test_a_turn_can_end_without_saying_why() -> None:
    """Not every provider reports a reason, and inventing one would lie."""
    assert RealtimeTurnEnd().finish_reason is None


def test_a_turn_end_carries_a_normalised_reason() -> None:
    assert RealtimeTurnEnd(finish_reason=FinishReason.STOP).finish_reason is (
        FinishReason.STOP
    )


def test_an_error_does_not_end_the_session_by_default() -> None:
    """Closing on every error would end a conversation over one bad frame."""
    assert RealtimeError(message="refused").recoverable


# Configuration


def test_a_session_is_text_only_unless_it_asks_for_audio() -> None:
    """Audio costs more and needs a voice. Opting in is the honest default."""
    assert RealtimeSessionConfig(model="m").modalities == [RealtimeModality.TEXT]


def test_the_input_rate_is_asked_of_the_session_not_assumed() -> None:
    """The providers disagree, and the wrong rate is silent: the model hears
    a voice at the wrong pitch rather than raising anything."""
    assert "input_sample_rate" in AbstractRealtimeSession.__abstractmethods__


# The seam itself


def test_the_seam_cannot_be_used_without_being_implemented() -> None:
    with pytest.raises(TypeError):
        AbstractRealtimeSession()  # type: ignore[abstract]


def test_a_conforming_session_satisfies_the_seam() -> None:
    """Pins the method set: adding one breaks this before it breaks an adapter."""

    class Conforming(AbstractRealtimeSession):
        @property
        def model(self) -> str:
            return "m"

        @property
        def input_sample_rate(self) -> int:
            return 24_000

        def get_service_name(self) -> str:
            return "conforming"

        async def aconnect(self) -> None: ...

        async def asend(self, value: RealtimeInput) -> None: ...

        def areceive(self) -> AsyncIterator[RealtimeOutput]:
            raise NotImplementedError

        async def asend_tool_result(self, call_id: str, result: str) -> None: ...

        async def ainterrupt(self) -> None: ...

        async def aclose(self) -> None: ...

    assert Conforming().get_service_name() == "conforming"


def test_the_seam_is_async_only() -> None:
    """A synchronous read would block a thread for the length of a conversation."""
    import inspect

    assert inspect.iscoroutinefunction(AbstractRealtimeSession.asend)
    assert inspect.iscoroutinefunction(AbstractRealtimeSession.aconnect)
    assert inspect.iscoroutinefunction(AbstractRealtimeSession.aclose)


def test_a_tool_result_is_answered_by_call_id() -> None:
    """A model may ask for several at once; position does not identify them."""
    import inspect

    assert (
        "call_id"
        in inspect.signature(AbstractRealtimeSession.asend_tool_result).parameters
    )
