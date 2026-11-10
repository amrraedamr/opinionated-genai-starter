"""OpenAI's realtime API, behind the provider-neutral session seam."""

import base64
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Final

from base.models.definitions.model import FinishReason, ProviderType
from base.models.definitions.realtime_session import (
    OPENAI_INPUT_SAMPLE_RATE_HZ,
    OPENAI_TRANSCRIPTION_MODEL,
    RealtimeModality,
)
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
from base.services.tool_calling import Toolbox
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

_TEXT_DELTA: Final = "response.output_text.delta"
_AUDIO_DELTA: Final = "response.output_audio.delta"
_ASSISTANT_TRANSCRIPT_DELTA: Final = "response.output_audio_transcript.delta"
_USER_TRANSCRIPT_DELTA: Final = "conversation.item.input_audio_transcription.delta"
_FUNCTION_CALL_DONE: Final = "response.function_call_arguments.done"
_RESPONSE_DONE: Final = "response.done"
_ERROR: Final = "error"
_CANCEL_NOT_ACTIVE: Final = "response_cancel_not_active"
"""Raised when a cancel lands after the turn already ended; harmless."""

_FINISH_REASONS: Final = {
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
        toolbox: Toolbox | None = None,
        provider: ProviderType = ProviderType.OPENAI,
    ) -> None:
        self._session = session
        self._config = config
        self._provider = provider
        self._toolbox = toolbox if toolbox is not None else Toolbox()
        self._awaiting: set[str] = set()
        self._logger = ConsoleLoggerService()
        self._manager: "AsyncRealtimeConnectionManager | None" = None
        self._connection: AsyncRealtimeConnection | None = None

    @property
    def model(self) -> str:
        return self._session.model

    @property
    def input_sample_rate(self) -> int:
        return OPENAI_INPUT_SAMPLE_RATE_HZ

    def get_service_name(self) -> str:
        return "openai_realtime"

    @property
    def config(self) -> ModelConfigurations:
        """Resolved on connect for the provider this session was built for."""
        if self._config is None:
            self._config = ModelConfigurations.get_default_for_provider(self._provider)
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
            "audio": {
                "input": {"transcription": {"model": OPENAI_TRANSCRIPTION_MODEL}}
            },
        }
        if self._session.instructions:
            payload["instructions"] = self._session.instructions
        if self._session.voice and RealtimeModality.AUDIO in self._session.modalities:
            payload["audio"]["output"] = {"voice": self._session.voice}
        if self._toolbox.tools:
            payload["tools"] = self._tools()
        return payload

    def _tools(self) -> "list[RealtimeFunctionToolParam]":
        """The toolbox in the realtime API's flat function shape."""
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.json_schema,
            }
            for tool in self._toolbox.tools
        ]

    async def asend(self, value: RealtimeInput) -> None:
        """Pass input to the model; only text is followed by a response request."""
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
        """Events as the model produces them, in our vocabulary; untranslated events are dropped."""
        connection = self._require_connection()
        async for event in connection:
            translated = self._translate(event)
            if translated is not None:
                yield translated

    def _translate(self, event: "RealtimeServerEvent") -> RealtimeOutput | None:
        kind = getattr(event, "type", None)
        if kind in (_TEXT_DELTA, _ASSISTANT_TRANSCRIPT_DELTA):
            return RealtimeTextDelta(text=event.delta)
        if kind == _AUDIO_DELTA:
            return RealtimeAudioDelta(audio=base64.b64decode(event.delta))
        if kind == _USER_TRANSCRIPT_DELTA:
            return RealtimeTranscriptDelta(text=event.delta) if event.delta else None
        if kind == _FUNCTION_CALL_DONE:
            self._awaiting.add(event.call_id)
            return RealtimeToolCall(
                call=StoredToolCall(
                    id=event.call_id,
                    function=StoredToolCallFunction(
                        name=event.name, arguments=event.arguments or "{}"
                    ),
                )
            )
        if kind == _RESPONSE_DONE:
            return self._turn_end(event)
        if kind == _ERROR:
            if event.error.code == _CANCEL_NOT_ACTIVE:
                return None
            return RealtimeError(message=event.error.message, recoverable=True)
        return None

    @staticmethod
    def _turn_end(event: "ResponseDoneEvent") -> RealtimeTurnEnd:
        response = event.response
        usage = response.usage
        return RealtimeTurnEnd(
            finish_reason=_FINISH_REASONS.get(
                response.status or "", FinishReason.OTHER
            ),
            usage=ProviderUsage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cached_input_tokens=(
                    usage.input_token_details.cached_tokens or 0
                    if usage.input_token_details
                    else 0
                ),
            )
            if usage
            else None,
        )

    async def asend_tool_result(self, call_id: str, result: str) -> None:
        """Answer a tool; once every call of the turn is answered, ask the model to carry on."""
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
        self._awaiting.discard(call_id)
        if not self._awaiting:
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
