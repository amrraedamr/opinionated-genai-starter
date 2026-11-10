"""What goes into a realtime session and what comes back out."""

import base64
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, PlainSerializer

from base.models.definitions.model import FinishReason
from base.models.definitions.realtime_session import (
    RealtimeEventType,
    RealtimeModality,
)
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredToolCall
from base.models.view_models.usage import ProviderUsage

AudioBytes = Annotated[
    bytes,
    BeforeValidator(
        lambda v: base64.b64decode(v, validate=True) if isinstance(v, str) else v
    ),
    PlainSerializer(
        lambda v: base64.b64encode(v).decode(), return_type=str, when_used="json"
    ),
]
"""Audio as bytes in Python and base64 in JSON."""


class RealtimeSessionConfig(BaseModel):
    """What a session is for, fixed when it opens."""

    model: str
    modalities: list[RealtimeModality] = [RealtimeModality.TEXT]
    instructions: str | None = None
    voice: str | None = None
    """Named per provider. Meaningless without `AUDIO` among the modalities."""


class RealtimeTextInput(BaseModel):
    """Something the user typed."""

    kind: Literal["text"] = "text"
    text: str


class RealtimeAudioInput(BaseModel):
    """A chunk of microphone audio, PCM at the agreed sample rate."""

    kind: Literal["audio"] = "audio"
    audio: AudioBytes


RealtimeInput = Annotated[
    RealtimeTextInput | RealtimeAudioInput, Field(discriminator="kind")
]


class RealtimeTextDelta(BaseModel):
    """Part of a written reply."""

    type: Literal[RealtimeEventType.TEXT_DELTA] = RealtimeEventType.TEXT_DELTA
    text: str


class RealtimeAudioDelta(BaseModel):
    """A chunk of spoken reply."""

    type: Literal[RealtimeEventType.AUDIO_DELTA] = RealtimeEventType.AUDIO_DELTA
    audio: AudioBytes


class RealtimeTranscriptDelta(BaseModel):
    """Part of what the model heard the user say."""

    type: Literal[RealtimeEventType.TRANSCRIPT_DELTA] = (
        RealtimeEventType.TRANSCRIPT_DELTA
    )
    text: str


class RealtimeToolCall(BaseModel):
    """A tool the model wants run before it carries on."""

    type: Literal[RealtimeEventType.TOOL_CALL] = RealtimeEventType.TOOL_CALL
    call: StoredToolCall


class RealtimeTurnEnd(BaseModel):
    """The model stopped. The session is still open."""

    type: Literal[RealtimeEventType.TURN_END] = RealtimeEventType.TURN_END
    finish_reason: FinishReason | None = None
    usage: ProviderUsage | None = None
    """Per turn, not per session."""


class RealtimeError(BaseModel):
    """The provider refused or failed."""

    type: Literal[RealtimeEventType.ERROR] = RealtimeEventType.ERROR
    message: str
    recoverable: bool = True
    """Whether the session is still usable after this error."""


RealtimeOutput = Annotated[
    RealtimeTextDelta
    | RealtimeAudioDelta
    | RealtimeTranscriptDelta
    | RealtimeToolCall
    | RealtimeTurnEnd
    | RealtimeError,
    Field(discriminator="type"),
]


__all__ = [
    "AudioBytes",
    "RealtimeAudioDelta",
    "RealtimeAudioInput",
    "RealtimeError",
    "RealtimeInput",
    "RealtimeOutput",
    "RealtimeSessionConfig",
    "RealtimeTextDelta",
    "RealtimeTextInput",
    "RealtimeToolCall",
    "RealtimeTranscriptDelta",
    "RealtimeTurnEnd",
]
