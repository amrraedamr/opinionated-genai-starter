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
    BeforeValidator(lambda v: base64.b64decode(v) if isinstance(v, str) else v),
    PlainSerializer(
        lambda v: base64.b64encode(v).decode(), return_type=str, when_used="json"
    ),
]
"""Audio as bytes in Python and base64 on the wire.

Plain `bytes` cannot be serialised to JSON: PCM is arbitrary, pydantic decodes
it as UTF-8, and the first sample above 0x7f raises. These frames cross a
websocket, which carries text. Pydantic's own `Base64Bytes` decodes on the way
in rather than encoding on the way out, which is the opposite of what a caller
holding raw samples needs.
"""


class RealtimeSessionConfig(BaseModel):
    """What a session is for, stated before it opens.

    Fixed at connection time because that is the only time a provider accepts
    it: a realtime session negotiates once, and changing its voice or its
    tools afterwards means opening a new one.
    """

    model: str
    modalities: list[RealtimeModality] = [RealtimeModality.TEXT]
    instructions: str | None = None
    voice: str | None = None
    """Named per provider. Meaningless without `AUDIO` among the modalities."""

    tool_names: list[str] = []
    """Resolved against `Toolbox`, so a session cannot reach a tool the
    service did not register."""


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
"""Typed and audio input are different enough to be separate shapes: one is
complete when it arrives, the other is a fragment of a stream that only means
anything in order.
"""


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
    """Per turn, not per session: a session has no total until it closes, and
    a caller charging for it cannot wait that long."""


class RealtimeError(BaseModel):
    """The provider refused or failed."""

    type: Literal[RealtimeEventType.ERROR] = RealtimeEventType.ERROR
    message: str
    recoverable: bool = True
    """Whether the session is still usable. A refused turn is not a dead
    socket, and closing on every error would end a conversation over one
    oversized frame."""


RealtimeOutput = Annotated[
    RealtimeTextDelta
    | RealtimeAudioDelta
    | RealtimeTranscriptDelta
    | RealtimeToolCall
    | RealtimeTurnEnd
    | RealtimeError,
    Field(discriminator="type"),
]
"""A union rather than one model with everything optional, so a caller that
handles every member is provably exhaustive, and a text delta cannot be
constructed holding audio.
"""


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
