"""A model session that stays open, and what travels through it."""

from enum import StrEnum
from typing import Final

OPENAI_INPUT_SAMPLE_RATE_HZ: Final = 24_000
GEMINI_INPUT_SAMPLE_RATE_HZ: Final = 16_000

OPENAI_TRANSCRIPTION_MODEL: Final = "whisper-1"
"""Requested on every OpenAI session; without it user-transcript deltas never arrive."""

REALTIME_USAGE_LABEL: Final = "realtime"


class RealtimeModality(StrEnum):
    """What a session is allowed to produce."""

    TEXT = "text"
    AUDIO = "audio"


class RealtimeEventType(StrEnum):
    """What arrived from the model."""

    TEXT_DELTA = "text_delta"

    AUDIO_DELTA = "audio_delta"
    """A chunk of spoken reply, in the session's agreed PCM format."""

    TRANSCRIPT_DELTA = "transcript_delta"
    """What the model heard the user say, not the model's own words."""

    TOOL_CALL = "tool_call"
    """The model wants a tool run before it continues. Mid-turn, not after."""

    TURN_END = "turn_end"
    """The model finished speaking. The session stays open."""

    ERROR = "error"
    """The provider refused or failed. May or may not end the session."""


__all__ = [
    "GEMINI_INPUT_SAMPLE_RATE_HZ",
    "OPENAI_INPUT_SAMPLE_RATE_HZ",
    "OPENAI_TRANSCRIPTION_MODEL",
    "REALTIME_USAGE_LABEL",
    "RealtimeEventType",
    "RealtimeModality",
]
