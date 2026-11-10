"""Gemini's Live API, behind the provider-neutral session seam."""

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Final

from base.models.definitions.model import FinishReason, ProviderType
from base.models.definitions.realtime import RealtimeCloseCode
from base.models.definitions.realtime_session import (
    GEMINI_INPUT_SAMPLE_RATE_HZ,
    RealtimeModality,
)
from base.models.view_models.chat_history import StoredToolCall, StoredToolCallFunction
from base.models.view_models.model import ModelConfigurations
from base.models.view_models.realtime_session import (
    RealtimeAudioDelta,
    RealtimeAudioInput,
    RealtimeInput,
    RealtimeOutput,
    RealtimeSessionConfig,
    RealtimeTextDelta,
    RealtimeToolCall,
    RealtimeTranscriptDelta,
    RealtimeTurnEnd,
)
from base.models.view_models.usage import ProviderUsage
from base.services.llm.gemini.adapter import GeminiContentAdapter
from base.services.llm.realtime import AbstractRealtimeSession
from base.services.logging.console import ConsoleLoggerService
from base.services.tool_calling import Toolbox
from base.tools.errors import MissingExtraError

try:
    from google.genai import Client
    from google.genai.errors import APIError
    from google.genai.types import (
        AudioTranscriptionConfig,
        Blob,
        Content,
        FunctionDeclarationDict,
        FunctionResponse,
        HttpOptions,
        LiveConnectConfig,
        Part,
        PrebuiltVoiceConfig,
        SpeechConfig,
        VoiceConfig,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "gemini", "google-genai", feature="GeminiLiveSession"
    ) from exc

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from google.genai.live import AsyncSession
    from google.genai.types import (
        Content as ModelTurn,
    )
    from google.genai.types import (
        LiveServerContent,
        LiveServerMessage,
    )

_INPUT_MIME_TYPE: Final = f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE_HZ}"

_MODALITIES: Final = {
    RealtimeModality.TEXT: "TEXT",
    RealtimeModality.AUDIO: "AUDIO",
}


class GeminiLiveSession(AbstractRealtimeSession):
    """One open conversation with a Gemini Live model."""

    def __init__(
        self,
        session: RealtimeSessionConfig,
        config: ModelConfigurations | None = None,
        toolbox: Toolbox | None = None,
    ) -> None:
        self._session = session
        self._config = config
        self._toolbox = toolbox if toolbox is not None else Toolbox()
        self._logger = ConsoleLoggerService()
        self._manager: "AbstractAsyncContextManager[AsyncSession] | None" = None
        self._live: AsyncSession | None = None
        self._tool_names: dict[str, str] = {}
        """Function name per outstanding call id; Gemini needs it to accept a result."""

    @property
    def model(self) -> str:
        return self._session.model

    @property
    def input_sample_rate(self) -> int:
        return GEMINI_INPUT_SAMPLE_RATE_HZ

    def get_service_name(self) -> str:
        return "gemini_live"

    @property
    def config(self) -> ModelConfigurations:
        """Resolved on connect, not construction."""
        if self._config is None:
            self._config = ModelConfigurations.get_default_for_provider(
                ProviderType.GEMINI
            )
        return self._config

    async def aconnect(self) -> None:
        """Open the session. Gemini negotiates everything up front."""
        client = Client(
            api_key=self.config.api_key,
            http_options=HttpOptions(base_url=self.config.api_host)
            if self.config.api_host
            else None,
        )
        self._manager = client.aio.live.connect(
            model=self._session.model, config=self._connect_config()
        )
        self._live = await self._manager.__aenter__()

    def _connect_config(self) -> LiveConnectConfig:
        """What this session negotiates, in Gemini's spelling, with both transcriptions on."""
        config = LiveConnectConfig(
            response_modalities=[_MODALITIES[m] for m in self._session.modalities],
            input_audio_transcription=AudioTranscriptionConfig(),
            output_audio_transcription=AudioTranscriptionConfig(),
        )
        if self._session.instructions:
            config.system_instruction = self._session.instructions
        if self._session.voice and RealtimeModality.AUDIO in self._session.modalities:
            config.speech_config = SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=self._session.voice
                    )
                )
            )
        if self._toolbox.tools:
            config.tools = [{"function_declarations": self._declarations()}]
        return config

    def _declarations(self) -> "list[FunctionDeclarationDict]":
        """The toolbox as Gemini function declarations, in the schema subset it accepts."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": GeminiContentAdapter.to_schema(tool.json_schema),
            }
            for tool in self._toolbox.tools
        ]

    async def asend(self, value: RealtimeInput) -> None:
        """Pass input to the model: text as a complete turn, audio as a stream."""
        live = self._require_live()
        if isinstance(value, RealtimeAudioInput):
            await live.send_realtime_input(
                audio=Blob(data=value.audio, mime_type=_INPUT_MIME_TYPE)
            )
            return
        await live.send_client_content(
            turns=Content(role="user", parts=[Part(text=value.text)]),
            turn_complete=True,
        )

    async def areceive(self) -> AsyncIterator[RealtimeOutput]:
        """Events across every turn; the SDK's `receive` ends after one."""
        live = self._require_live()
        while True:
            try:
                async for message in live.receive():
                    for event in self._translate(message):
                        yield event
            except APIError as error:
                if error.code == RealtimeCloseCode.NORMAL:
                    return
                raise

    def _translate(self, message: "LiveServerMessage") -> list[RealtimeOutput]:
        events: list[RealtimeOutput] = []
        content = message.server_content

        if content is not None:
            if content.input_transcription and content.input_transcription.text:
                events.append(
                    RealtimeTranscriptDelta(text=content.input_transcription.text)
                )
            if content.output_transcription and content.output_transcription.text:
                events.append(RealtimeTextDelta(text=content.output_transcription.text))
            if content.model_turn:
                events.extend(self._parts_of(content.model_turn))

        if message.tool_call:
            for call in message.tool_call.function_calls or []:
                identifier = call.id or call.name or ""
                self._tool_names[identifier] = call.name or ""
                events.append(
                    RealtimeToolCall(
                        call=StoredToolCall(
                            id=identifier,
                            function=StoredToolCallFunction(
                                name=call.name or "",
                                arguments=json.dumps(call.args or {}),
                            ),
                        )
                    )
                )

        if content is not None and content.turn_complete:
            events.append(self._turn_end(message, content))
        return events

    @staticmethod
    def _parts_of(model_turn: "ModelTurn") -> list[RealtimeOutput]:
        """Text and audio arrive as parts of one turn, not separate events."""
        events: list[RealtimeOutput] = []
        for part in model_turn.parts or []:
            if part.text:
                events.append(RealtimeTextDelta(text=part.text))
            if part.inline_data and part.inline_data.data:
                events.append(RealtimeAudioDelta(audio=part.inline_data.data))
        return events

    @staticmethod
    def _turn_end(
        message: "LiveServerMessage", content: "LiveServerContent"
    ) -> RealtimeTurnEnd:
        usage = message.usage_metadata
        return RealtimeTurnEnd(
            finish_reason=FinishReason.OTHER
            if content.interrupted
            else FinishReason.STOP,
            usage=ProviderUsage(
                input_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.response_token_count or 0,
                cached_input_tokens=usage.cached_content_token_count or 0,
            )
            if usage
            else None,
        )

    async def asend_tool_result(self, call_id: str, result: str) -> None:
        """Answer a tool the model asked for; Gemini resumes without being asked."""
        live = self._require_live()
        await live.send_tool_response(
            function_responses=[
                FunctionResponse(
                    id=call_id,
                    name=self._tool_names.pop(call_id, call_id),
                    response={"result": result},
                )
            ]
        )

    async def ainterrupt(self) -> None:
        """A no-op: with automatic activity detection, new input interrupts."""
        self._require_live()

    async def aclose(self) -> None:
        """End the session. Quiet if it is already closed."""
        if self._manager is None:
            return
        manager, self._manager, self._live = self._manager, None, None
        try:
            await manager.__aexit__(None, None, None)
        except Exception as error:
            self._logger.warning(f"Gemini live session closed uncleanly: {error}")

    def _require_live(self) -> "AsyncSession":
        if self._live is None:
            raise RuntimeError("Realtime session used before aconnect")
        return self._live


__all__ = ["GeminiLiveSession"]
