"""Gemini's Live API, behind the provider-neutral session seam.

Shaped differently from OpenAI's in ways that matter to the translation.

Gemini listens at 16 kHz while OpenAI listens at 24 kHz, and both speak at
24 kHz. Sending the wrong rate raises nothing: the model hears a voice at the
wrong pitch and answers a question nobody asked. The rate is declared on the
session so a caller resamples against what it actually opened.

Text and audio are not separate events here. Both arrive as parts of a model
turn, and the part says which it is.

There is no cancel message. Gemini stops on its own when fresh input arrives
and reports it afterwards as an interruption, so interrupting is a matter of
telling it the user has started talking rather than asking it to stop.
"""

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Final

from base.models.definitions.model import FinishReason, ProviderType
from base.models.definitions.realtime import RealtimeCloseCode
from base.models.definitions.realtime_session import RealtimeModality
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
from base.services.llm.realtime import AbstractRealtimeSession
from base.services.logging.console import ConsoleLoggerService
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

INPUT_SAMPLE_RATE_HZ: Final = 16_000
"""What Gemini listens at. OpenAI listens at 24 kHz; neither resamples."""

INPUT_MIME_TYPE: Final = f"audio/pcm;rate={INPUT_SAMPLE_RATE_HZ}"

MODALITIES: Final = {
    RealtimeModality.TEXT: "TEXT",
    RealtimeModality.AUDIO: "AUDIO",
}


class GeminiLiveSession(AbstractRealtimeSession):
    """One open conversation with a Gemini Live model."""

    def __init__(
        self,
        session: RealtimeSessionConfig,
        config: ModelConfigurations | None = None,
        tool_schemas: "list[FunctionDeclarationDict] | None" = None,
    ) -> None:
        self._session = session
        self._config = config
        self._tool_schemas = tool_schemas or []
        self._logger = ConsoleLoggerService()
        self._manager: "AbstractAsyncContextManager[AsyncSession] | None" = None
        self._live: AsyncSession | None = None
        self._tool_names: dict[str, str] = {}
        """Which function each outstanding call belongs to.

        Gemini needs the function name to accept a result; OpenAI needs only
        the call id. Remembering it here keeps that difference out of the
        seam, which would otherwise carry a field only one provider reads.
        """

    @property
    def model(self) -> str:
        return self._session.model

    @property
    def input_sample_rate(self) -> int:
        return INPUT_SAMPLE_RATE_HZ

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
        """What this session negotiates, in Gemini's spelling.

        Both transcriptions are asked for. They are off by default, and
        without them a caller can neither show what was heard nor caption an
        audio reply, since in an audio session the model's words arrive only
        as its own transcript.
        """
        config = LiveConnectConfig(
            response_modalities=[MODALITIES[m] for m in self._session.modalities],
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
        if self._tool_schemas:
            config.tools = [{"function_declarations": self._tool_schemas}]
        return config

    async def asend(self, value: RealtimeInput) -> None:
        """Pass input to the model.

        Text is sent as a complete turn, audio as a stream whose end the
        server detects. Marking every audio chunk as a finished turn would
        start a turn for each packet of microphone input.
        """
        live = self._require_live()
        if isinstance(value, RealtimeAudioInput):
            await live.send_realtime_input(
                audio=Blob(data=value.audio, mime_type=INPUT_MIME_TYPE)
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
            )
            if usage
            else None,
        )

    async def asend_tool_result(self, call_id: str, result: str) -> None:
        """Answer a tool the model asked for.

        No response request follows. Gemini resumes on the tool response
        itself, where OpenAI needs to be asked again.
        """
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
