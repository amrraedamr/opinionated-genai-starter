"""OpenAI's realtime API, behind the provider-neutral session seam.

Two things here are easy to get wrong and quiet when they are.

The event names moved. What older documentation calls `response.text.delta`
and `response.audio.delta` are now `response.output_text.delta` and
`response.output_audio.delta`. An adapter matching the old names compiles,
connects, and yields nothing at all.

There are two transcript events and they describe different people.
`conversation.item.input_audio_transcription.delta` is what the model heard
the user say; `response.output_audio_transcript.delta` is what the model
itself is saying, which in an audio session is the only place its words
arrive as text. Treating them alike puts the user's sentence in the model's
mouth.
"""

import base64
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Final

from base.models.definitions.model import FinishReason, ProviderType
from base.models.definitions.realtime_session import RealtimeModality
from base.models.view_models.chat_history import StoredToolCall, StoredToolCallFunction
from base.models.view_models.model import ModelConfigurations
from base.models.view_models.realtime_session import (
    RealtimeAudioDelta,
    RealtimeAudioInput,
    RealtimeError,
    RealtimeInput,
    RealtimeOutput,
    RealtimeSessionConfig,
    RealtimeTextDelta,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeTurnEnd,
)
from base.models.view_models.usage import ProviderUsage
from base.services.llm.realtime import AbstractRealtimeSession
from base.services.logging.console import ConsoleLoggerService
from base.tools.openai_clients import create_openai_clients

if TYPE_CHECKING:
    from openai.resources.realtime.realtime import (
        AsyncRealtimeConnection,
        AsyncRealtimeConnectionManager,
    )
    from openai.types.realtime import (
        RealtimeFunctionToolParam,
        RealtimeServerEvent,
        ResponseDoneEvent,
    )
    from openai.types.realtime.realtime_session_create_request_param import (
        RealtimeSessionCreateRequestParam,
    )

TEXT_DELTA: Final = "response.output_text.delta"
AUDIO_DELTA: Final = "response.output_audio.delta"
ASSISTANT_TRANSCRIPT_DELTA: Final = "response.output_audio_transcript.delta"
USER_TRANSCRIPT_DELTA: Final = "conversation.item.input_audio_transcription.delta"
FUNCTION_CALL_DONE: Final = "response.function_call_arguments.done"
RESPONSE_DONE: Final = "response.done"
ERROR: Final = "error"

INPUT_SAMPLE_RATE_HZ: Final = 24_000
"""What OpenAI listens at. Gemini listens at 16 kHz; neither resamples."""

TRANSCRIPTION_MODEL: Final = "whisper-1"
"""Input transcription is off unless a session asks for it, and without it
`conversation.item.input_audio_transcription.delta` never arrives — the seam
would declare an event the adapter could never produce.

Asked for on every session, not only audio-replying ones. It describes what
arrives, not what leaves, and speaking to a session that answers in text is
the ordinary case. Gating it on the output modality would leave exactly that
caller unable to show the user what was heard."""

FINISH_REASONS: Final = {
    "completed": FinishReason.STOP,
    "incomplete": FinishReason.LENGTH,
    "cancelled": FinishReason.OTHER,
    "failed": FinishReason.OTHER,
}


class OpenAIRealtimeSession(AbstractRealtimeSession):
    """One open conversation with an OpenAI realtime model."""

    def __init__(
        self,
        session: RealtimeSessionConfig,
        config: ModelConfigurations | None = None,
        tool_schemas: "list[RealtimeFunctionToolParam] | None" = None,
    ) -> None:
        self._session = session
        self._config = config
        self._tool_schemas = tool_schemas or []
        self._logger = ConsoleLoggerService()
        self._manager: "AsyncRealtimeConnectionManager | None" = None
        self._connection: AsyncRealtimeConnection | None = None

    @property
    def model(self) -> str:
        return self._session.model

    @property
    def input_sample_rate(self) -> int:
        return INPUT_SAMPLE_RATE_HZ

    def get_service_name(self) -> str:
        return "openai_realtime"

    @property
    def config(self) -> ModelConfigurations:
        """Resolved on connect, not construction.

        Settings are not registered when a session object is built, and
        reading them there would make holding one depend on the order a
        service happens to start things in.
        """
        if self._config is None:
            self._config = ModelConfigurations.get_default_for_provider(
                ProviderType.AZURE_OPENAI
            )
        return self._config

    async def aconnect(self) -> None:
        """Open the socket and state what the session is for."""
        _, client = create_openai_clients(
            api_key=self.config.api_key,
            api_host=self.config.api_host,
            api_type=self.config.api_type,
        )
        self._manager = client.realtime.connect(model=self._session.model)
        self._connection = await self._manager.__aenter__()
        await self._connection.send(
            {"type": "session.update", "session": self._session_payload()}
        )

    def _session_payload(self) -> "RealtimeSessionCreateRequestParam":
        """What this session negotiates, in OpenAI's spelling."""
        payload: "RealtimeSessionCreateRequestParam" = {
            "type": "realtime",
            "output_modalities": [m.value for m in self._session.modalities],
            "audio": {"input": {"transcription": {"model": TRANSCRIPTION_MODEL}}},
        }
        if self._session.instructions:
            payload["instructions"] = self._session.instructions
        if self._session.voice and RealtimeModality.AUDIO in self._session.modalities:
            payload["audio"]["output"] = {"voice": self._session.voice}
        if self._tool_schemas:
            payload["tools"] = self._tool_schemas
        return payload

    async def asend(self, value: RealtimeInput) -> None:
        """Pass input to the model.

        Text is committed with an explicit response request; audio is not.
        Audio arrives as a stream the server decides the end of, and asking
        for a response on every chunk would start a turn per packet.
        """
        connection = self._require_connection()
        if isinstance(value, RealtimeAudioInput):
            await connection.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(value.audio).decode(),
                }
            )
            return

        await connection.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": value.text}],
                },
            }
        )
        await connection.send({"type": "response.create"})

    async def areceive(self) -> AsyncIterator[RealtimeOutput]:
        """Events as the model produces them, in our vocabulary.

        An event this adapter does not translate is dropped rather than
        surfaced. The provider emits a great deal of bookkeeping — item
        created, buffer committed, response in progress — and passing it
        through would make every caller filter the same noise.
        """
        connection = self._require_connection()
        async for event in connection:
            translated = self._translate(event)
            if translated is not None:
                yield translated

    def _translate(self, event: "RealtimeServerEvent") -> RealtimeOutput | None:
        kind = getattr(event, "type", None)
        if kind in (TEXT_DELTA, ASSISTANT_TRANSCRIPT_DELTA):
            return RealtimeTextDelta(text=event.delta)
        if kind == AUDIO_DELTA:
            return RealtimeAudioDelta(audio=base64.b64decode(event.delta))
        if kind == USER_TRANSCRIPT_DELTA:
            return RealtimeTranscriptDelta(text=event.delta)
        if kind == FUNCTION_CALL_DONE:
            return RealtimeToolCall(
                call=StoredToolCall(
                    id=event.call_id,
                    function=StoredToolCallFunction(
                        name=event.name, arguments=event.arguments or "{}"
                    ),
                )
            )
        if kind == RESPONSE_DONE:
            return self._turn_end(event)
        if kind == ERROR:
            return RealtimeError(message=str(event.error), recoverable=True)
        return None

    @staticmethod
    def _turn_end(event: "ResponseDoneEvent") -> RealtimeTurnEnd:
        response = event.response
        usage = response.usage
        return RealtimeTurnEnd(
            finish_reason=FINISH_REASONS.get(response.status or "", FinishReason.OTHER),
            usage=ProviderUsage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
            )
            if usage
            else None,
        )

    async def asend_tool_result(self, call_id: str, result: str) -> None:
        """Answer a tool and let the model carry on.

        The result is submitted and a response asked for, because the model
        stopped when it asked; without the second message it waits forever.
        """
        connection = self._require_connection()
        await connection.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                },
            }
        )
        await connection.send({"type": "response.create"})

    async def ainterrupt(self) -> None:
        """Cancel the turn in progress because the user started talking."""
        connection = self._require_connection()
        await connection.send({"type": "response.cancel"})

    async def aclose(self) -> None:
        """Close the socket. Quiet if it is already closed."""
        if self._manager is None:
            return
        manager, self._manager, self._connection = self._manager, None, None
        try:
            await manager.__aexit__(None, None, None)
        except Exception as error:
            self._logger.warning(f"Realtime session closed uncleanly: {error}")

    def _require_connection(self) -> "AsyncRealtimeConnection":
        if self._connection is None:
            raise RuntimeError("Realtime session used before aconnect")
        return self._connection


__all__ = ["OpenAIRealtimeSession"]
