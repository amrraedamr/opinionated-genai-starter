"""A model session that stays open, and what travels through it.

Separate from the request/response vocabulary because a realtime session is
not a call. There is no single moment the model is finished: it emits deltas,
asks for tools mid-turn, and can be cut off halfway through speaking. Forcing
that into a prompt-in/result-out shape would misdescribe both.
"""

from enum import StrEnum
from typing import Final

AUDIO_SAMPLE_RATE_HZ: Final = 24_000
"""What both OpenAI and Gemini expect for PCM audio. Stated once so an adapter
resamples rather than each caller guessing."""


class RealtimeModality(StrEnum):
    """What a session is allowed to produce."""

    TEXT = "text"
    AUDIO = "audio"


class RealtimeEventType(StrEnum):
    """What arrived from the model.

    Branch on this rather than on whichever field happens to be populated: a
    turn can end with no text at all, and an empty delta is not the same event
    as the end of a turn.
    """

    TEXT_DELTA = "text_delta"
    """Part of a written reply. Arrives in fragments, not sentences."""

    AUDIO_DELTA = "audio_delta"
    """A chunk of spoken reply, in the session's agreed PCM format."""

    TRANSCRIPT_DELTA = "transcript_delta"
    """What the model heard the user say.

    Distinct from text output: it is the input read back, and showing it as
    the assistant's words would attribute the user's sentence to the model.
    """

    TOOL_CALL = "tool_call"
    """The model wants a tool run before it continues. Mid-turn, not after."""

    TURN_END = "turn_end"
    """The model finished speaking. The session stays open."""

    ERROR = "error"
    """The provider refused or failed. May or may not end the session."""


__all__ = [
    "AUDIO_SAMPLE_RATE_HZ",
    "RealtimeEventType",
    "RealtimeModality",
]
