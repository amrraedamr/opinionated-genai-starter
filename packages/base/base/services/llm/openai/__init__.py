import time
from collections.abc import Sequence
from typing import Any, cast
from uuid import uuid4

import httpx

from base.tools.errors import MissingExtraError

try:
    from openai import (
        AsyncOpenAI,
        AsyncStream,
        OpenAI,
        OpenAIError,
        Stream,
    )
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        ChatCompletionStreamOptionsParam,
        ParsedChatCompletion,
    )
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses import (
        ParsedResponse,
        Response,
        ResponseCompletedEvent,
        ResponseCreatedEvent,
        ResponseIncompleteEvent,
        ResponseInputParam,
        ResponseStreamEvent,
        ResponseTextDeltaEvent,
        ResponseUsage,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="OpenAIService") from exc

from base.models.definitions.model import (
    FinishReason,
    ProviderType,
    ReasoningEffort,
)
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredMessage, StoredToolCall
from base.models.view_models.debug_trace import (
    DebugCall,
    LLMResponseContent,
)
from base.models.view_models.exceptions.llm import (
    LLMConfigError,
    LLMContentFilterError,
    LLMError,
    LLMIncompleteError,
    LLMStreamError,
)
from base.models.view_models.llm_result import (
    LLMResult,
    LLMUsageCounts,
    ParsedResult,
)
from base.models.view_models.model import ModelConfigurations
from base.models.view_models.openai_results import (
    CompletionResult,
    LLMPrompt,
    LLMUsage,
    ParsedCompletionResult,
    ParsedResponseResult,
    ResponseResult,
)
from base.models.view_models.prompt import RenderedPrompt
from base.models.view_models.streams import (
    StreamEnd,
    StreamError,
    StreamMessage,
    StreamStart,
)
from base.services.common.meta import KeyedSingletonABC
from base.services.configuration import get_settings
from base.services.configuration.protocols import SupportsLLMConfig
from base.services.error_handling.llm.openai import OpenAIErrorHandler
from base.services.execution_context import ExecutionContext
from base.services.llm import AbstractLLMService
from base.services.llm.openai.adapter import OpenAIMessageAdapter
from base.services.logging.console import ConsoleLoggerService
from base.services.tool_calling import Tool
from base.services.usage import UsageTracker
from base.tools.openai_clients import create_openai_clients
from base.tools.stream import StreamIterator

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class OpenAIService(AbstractLLMService, metaclass=KeyedSingletonABC):
    """
    OpenAI-compatible LLM service.

    Works with any provider following the OpenAI API spec:
    OpenAI, Azure OpenAI (v1 surface), vLLM, Ollama, LiteLLM, Together, Groq, etc.

    Error handling is delegated to OpenAIErrorHandler. All errors raised from
    public methods are LLMError subclasses — callers should never see raw
    OpenAI SDK exceptions.

    `auth` carries credentials the static `api_key` cannot express; see
    `base.clients.credentials`. It is not serializable, so a caller passing
    one must also pass an explicit `_key`. Such a credential is acquired in a
    coroutine, so an instance holding one is async-only and its sync methods
    refuse rather than letting the SDK reclassify the failure as retryable.
    """

    DEFAULT_TIMEOUT_TOTAL: float = 60.0
    DEFAULT_TIMEOUT_CONNECT: float = 5.0

    DEFAULT_TEMPERATURE: float = 0.3
    """Sampling temperature applied to every LLM call by default. Kept low
    (but non-zero) so routing / extraction stages are stable across runs
    while still allowing the answer agents some freedom to phrase
    naturally. Per-call overrides via ``temperature=`` kwarg still win."""

    def __init__(
        self,
        config: ModelConfigurations | None = None,
        auth: httpx.Auth | None = None,
        verify: bool = True,
    ) -> None:
        if not config:
            config = ModelConfigurations.get_default_for_provider(
                ProviderType.AZURE_OPENAI
            )

        if config.model_name is None:
            raise LLMConfigError("model_name is required")

        self._config: ModelConfigurations = config
        self._model: str = config.model_name
        self._supports_responses_api: bool = config.supports_responses_api
        self._auth: httpx.Auth | None = auth
        self._verify: bool = verify

        self._logger = ConsoleLoggerService()

        self._client, self._async_client = self._initialize_clients()

        self._error_handler: OpenAIErrorHandler = OpenAIErrorHandler(
            retries=config.api_retries,
            base_wait=config.wait_time,
        )

    @property
    def model(self) -> str:
        """Public read-only access to the model name (e.g. ``gpt-4o``)
        so callers can attach it to per-call usage records.
        """
        return self._model

    @property
    def _execution_context(self) -> ExecutionContext:
        """Resolve the current request's :class:`ExecutionContext`
        lazily. :class:`OpenAIService` is a :class:`KeyedSingleton`
        (cached per model config), but :class:`ExecutionContext` is
        per-request scoped via ``Stored``. A property dispatches every
        access to the current scope's instance; storing it in
        ``__init__`` would freeze the first request's context across
        all subsequent requests.
        """
        return ExecutionContext()

    def _apply_kwarg_defaults(self, kwargs: dict[str, object]) -> None:
        """Inject service-wide kwarg defaults into a public-method's
        ``**kwargs`` dict before it's forwarded to the OpenAI SDK.

        Uses ``setdefault`` semantics so per-call overrides win.
        Today: only ``temperature``.
        """
        kwargs.setdefault("temperature", self.DEFAULT_TEMPERATURE)

    def _reject_sync_call(self) -> None:
        """Refuse a sync call when credentials are acquired asynchronously.

        Letting it reach the SDK surfaces as `APIConnectionError`, which the
        retry policy treats as transient — so a permanent misconfiguration
        would be retried and then reported as a connection failure.
        """
        if self._auth is not None:
            raise LLMConfigError(
                "This OpenAIService authenticates with an asynchronous "
                "credential and has no synchronous surface. Use the "
                "`a`-prefixed methods."
            )

    def _initialize_clients(self) -> tuple[OpenAI, AsyncOpenAI]:
        return create_openai_clients(
            api_key=self._config.api_key,
            api_host=self._config.api_host,
            api_type=self._config.api_type,
            timeout_total=self.DEFAULT_TIMEOUT_TOTAL,
            timeout_connect=self.DEFAULT_TIMEOUT_CONNECT,
            auth=self._auth,
            verify=self._verify,
        )

    # Stream error helper

    @staticmethod
    def _build_stream_error(
        error: Exception,
        *,
        error_code: int,
        text: str | None = None,
        extra_detail: dict[str, object] | None = None,
    ) -> StreamError:
        """Construct a StreamError with consistent detail payload."""
        detail: dict[str, object] = {
            "exception_type": type(error).__name__,
            "exception_module": type(error).__module__,
        }
        if extra_detail:
            detail.update(extra_detail)
        return StreamError(
            error_code=error_code,
            text=text if text is not None else str(error),
            detail=detail,
        )

    # Completions

    def completions(
        self,
        messages: list[ChatCompletionMessageParam],
        stream_iterator: StreamIterator | None = None,
        **kwargs,
    ) -> CompletionResult:
        self._reject_sync_call()
        streaming: bool = bool(stream_iterator)
        label: str | None = kwargs.pop("label", None)
        self._apply_kwarg_defaults(kwargs)

        t0 = time.perf_counter()
        result: CompletionResult | None = None
        error: str | None = None
        try:
            response: ChatCompletion | Stream[ChatCompletionChunk] = (
                self._error_handler.run(
                    self._client.chat.completions.create,
                    model=self._model,
                    messages=messages,
                    stream=streaming,
                    stream_options=ChatCompletionStreamOptionsParam(include_usage=True)
                    if streaming
                    else None,
                    **kwargs,
                )
            )

            if streaming and isinstance(response, Stream):
                assert stream_iterator is not None
                result = self._stream_completions(response, stream_iterator)
                return result

            assert isinstance(response, ChatCompletion)
            result = CompletionResult(
                message=response.choices[0].message,
                usage=response.usage,
                finish_reason=response.choices[0].finish_reason,
            )
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=messages,
                response=(result.content if result else None),
                usage=(result.usage if result else None),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    async def acompletions(
        self,
        messages: list[ChatCompletionMessageParam],
        stream_iterator: StreamIterator | None = None,
        **kwargs,
    ) -> CompletionResult:
        streaming: bool = bool(stream_iterator)
        label: str | None = kwargs.pop("label", None)
        self._apply_kwarg_defaults(kwargs)

        t0 = time.perf_counter()
        result: CompletionResult | None = None
        error: str | None = None
        try:
            response: (
                ChatCompletion | AsyncStream[ChatCompletionChunk]
            ) = await self._error_handler.arun(
                self._async_client.chat.completions.create,
                model=self._model,
                messages=messages,
                stream=streaming,
                stream_options=ChatCompletionStreamOptionsParam(include_usage=True)
                if streaming
                else None,
                **kwargs,
            )

            if streaming and isinstance(response, AsyncStream):
                assert stream_iterator is not None
                result = await self._astream_completions(response, stream_iterator)
                return result

            assert isinstance(response, ChatCompletion)
            result = CompletionResult(
                message=response.choices[0].message,
                usage=response.usage,
                finish_reason=response.choices[0].finish_reason,
            )
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=messages,
                response=(result.content if result else None),
                usage=(result.usage if result else None),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    def _stream_completions(
        self,
        response: Stream[ChatCompletionChunk],
        stream_iterator: StreamIterator,
    ) -> CompletionResult:
        parts: list[str] = []
        started: bool = False
        result = CompletionResult(
            message=ChatCompletionMessage(content="", role="assistant"),
        )

        try:
            for chunk in response:
                if not started:
                    started = True
                    stream_iterator.send(StreamStart())

                for choice in chunk.choices:
                    if choice.delta and choice.delta.content:
                        parts.append(choice.delta.content)
                        stream_iterator.send(StreamMessage(data=choice.delta.content))
                    if choice.finish_reason:
                        result.finish_reason = choice.finish_reason

                if chunk.usage:
                    result.usage = chunk.usage

            stream_iterator.send(StreamEnd())

        except OpenAIError as e:
            self._logger.error(f"Error during streaming completion: {e}")
            stream_iterator.send(self._build_stream_error(e, error_code=500))
            raise LLMStreamError(
                f"Stream failed: {e}",
                partial_content="".join(parts) if parts else None,
                provider_error=e,
            ) from e

        result.message = ChatCompletionMessage(content="".join(parts), role="assistant")
        return result

    async def _astream_completions(
        self,
        response: AsyncStream[ChatCompletionChunk],
        stream_iterator: StreamIterator,
    ) -> CompletionResult:
        parts: list[str] = []
        started: bool = False
        result = CompletionResult(
            message=ChatCompletionMessage(content="", role="assistant"),
        )

        try:
            async for chunk in response:
                if not started:
                    started = True
                    await stream_iterator.asend(StreamStart())

                for choice in chunk.choices:
                    if choice.delta and choice.delta.content:
                        parts.append(choice.delta.content)
                        await stream_iterator.asend(
                            StreamMessage(data=choice.delta.content)
                        )
                    if choice.finish_reason:
                        result.finish_reason = choice.finish_reason

                if chunk.usage:
                    result.usage = chunk.usage

            await stream_iterator.asend(StreamEnd())

        except OpenAIError as e:
            self._logger.error(f"Error during streaming completion: {e}")
            await stream_iterator.asend(self._build_stream_error(e, error_code=500))
            raise LLMStreamError(
                f"Stream failed: {e}",
                partial_content="".join(parts) if parts else None,
                provider_error=e,
            ) from e

        result.message = ChatCompletionMessage(content="".join(parts), role="assistant")
        return result

    # Responses

    def responses(
        self,
        input_: ResponseInputParam,
        stream_iterator: StreamIterator | None = None,
        **kwargs,
    ) -> ResponseResult:
        self._reject_sync_call()
        streaming: bool = bool(stream_iterator)
        label: str | None = kwargs.pop("label", None)
        self._apply_kwarg_defaults(kwargs)

        t0 = time.perf_counter()
        result: ResponseResult | None = None
        error: str | None = None
        try:
            response: Response | Stream[ResponseStreamEvent] = self._error_handler.run(
                self._client.responses.create,
                model=self._model,
                input=input_,
                stream=streaming,
                **kwargs,
            )

            if streaming and isinstance(response, Stream):
                assert stream_iterator is not None
                result = self._stream_responses(response, stream_iterator)
                return result

            assert isinstance(response, Response)
            self._raise_if_incomplete(response)
            result = ResponseResult(response=response, usage=response.usage)
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=input_,
                response=(result.content if result else None),
                usage=(result.usage if result else None),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    async def aresponses(
        self,
        input_: ResponseInputParam,
        stream_iterator: StreamIterator | None = None,
        **kwargs,
    ) -> ResponseResult:
        streaming: bool = bool(stream_iterator)
        label: str | None = kwargs.pop("label", None)
        self._apply_kwarg_defaults(kwargs)

        t0 = time.perf_counter()
        result: ResponseResult | None = None
        error: str | None = None
        try:
            response: (
                Response | AsyncStream[ResponseStreamEvent]
            ) = await self._error_handler.arun(
                self._async_client.responses.create,
                model=self._model,
                input=input_,
                stream=streaming,
                **kwargs,
            )

            if streaming and isinstance(response, AsyncStream):
                assert stream_iterator is not None
                result = await self._astream_responses(response, stream_iterator)
                return result

            assert isinstance(response, Response)
            self._raise_if_incomplete(response)
            result = ResponseResult(response=response, usage=response.usage)
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=input_,
                response=(result.content if result else None),
                usage=(result.usage if result else None),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    @staticmethod
    def _raise_if_incomplete(response: Response) -> None:
        """Detect non-streaming Responses-API replies that were cut short
        (content filter, max-tokens, etc.) and surface them as
        ``LLMContentFilterError`` / ``LLMIncompleteError`` instead of
        returning the truncated body to callers.
        """
        if getattr(response, "status", None) != "incomplete":
            return
        details = getattr(response, "incomplete_details", None)
        if details and getattr(details, "reason", None) == "content_filter":
            raise LLMContentFilterError(
                "Response blocked by content filter",
                incomplete_details=details,
            )
        raise LLMIncompleteError(
            f"Response incomplete: {details}",
            incomplete_details=details,
        )

    def _stream_responses(
        self,
        response: Stream[ResponseStreamEvent],
        stream_iterator: StreamIterator,
    ) -> ResponseResult:
        _response: Response | None = None
        usage = None
        stream_started: bool = False

        try:
            for event in response:
                if isinstance(event, ResponseCreatedEvent) and not stream_started:
                    stream_started = True
                    stream_iterator.send(StreamStart())

                if isinstance(event, ResponseTextDeltaEvent):
                    stream_iterator.send(StreamMessage(data=event.delta))

                if isinstance(event, ResponseCompletedEvent):
                    if event.response.usage:
                        usage = event.response.usage
                    _response = event.response
                    stream_iterator.send(StreamEnd())
                    break

                if isinstance(event, ResponseIncompleteEvent):
                    details = event.response.incomplete_details
                    if details and details.reason == "content_filter":
                        raise LLMContentFilterError(
                            "Response blocked by content filter",
                            incomplete_details=details,
                        )
                    raise LLMIncompleteError(
                        f"Response incomplete: {details}",
                        incomplete_details=details,
                    )

        except LLMContentFilterError as e:
            stream_iterator.send(
                self._build_stream_error(
                    e,
                    error_code=400,
                    text="Response blocked by content filter",
                    extra_detail={"reason": "content_filter"},
                )
            )
            raise

        except LLMIncompleteError as e:
            stream_iterator.send(
                self._build_stream_error(
                    e,
                    error_code=499,
                    text="Response incomplete",
                    extra_detail={"reason": "incomplete"},
                )
            )
            raise

        except OpenAIError as e:
            self._logger.error(f"Error during streaming response: {e}")
            stream_iterator.send(self._build_stream_error(e, error_code=500))
            raise LLMStreamError(
                f"Stream failed: {e}",
                provider_error=e,
            ) from e

        assert _response is not None, "Stream ended without a completed response"
        return ResponseResult(response=_response, usage=usage)

    async def _astream_responses(
        self,
        response: AsyncStream[ResponseStreamEvent],
        stream_iterator: StreamIterator,
    ) -> ResponseResult:
        _response: Response | None = None
        usage = None
        stream_started: bool = False

        try:
            async for event in response:
                if isinstance(event, ResponseCreatedEvent) and not stream_started:
                    stream_started = True
                    await stream_iterator.asend(StreamStart())

                if isinstance(event, ResponseTextDeltaEvent):
                    await stream_iterator.asend(StreamMessage(data=event.delta))

                if isinstance(event, ResponseCompletedEvent):
                    if event.response.usage:
                        usage = event.response.usage
                    _response = event.response
                    await stream_iterator.asend(StreamEnd())
                    break

                if isinstance(event, ResponseIncompleteEvent):
                    details = event.response.incomplete_details
                    if details and details.reason == "content_filter":
                        raise LLMContentFilterError(
                            "Response blocked by content filter",
                            incomplete_details=details,
                        )
                    raise LLMIncompleteError(
                        f"Response incomplete: {details}",
                        incomplete_details=details,
                    )

        except LLMContentFilterError as e:
            await stream_iterator.asend(
                self._build_stream_error(
                    e,
                    error_code=400,
                    text="Response blocked by content filter",
                    extra_detail={"reason": "content_filter"},
                )
            )
            raise

        except LLMIncompleteError as e:
            await stream_iterator.asend(
                self._build_stream_error(
                    e,
                    error_code=499,
                    text="Response incomplete",
                    extra_detail={"reason": "incomplete"},
                )
            )
            raise

        except OpenAIError as e:
            self._logger.error(f"Error during streaming response: {e}")
            await stream_iterator.asend(self._build_stream_error(e, error_code=500))
            raise LLMStreamError(
                f"Stream failed: {e}",
                provider_error=e,
            ) from e

        assert _response is not None, "Stream ended without a completed response"
        return ResponseResult(response=_response, usage=usage)

    # Parse — Chat Completions

    def parse_completions[T: BaseModel](
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[T],
        **kwargs,
    ) -> ParsedCompletionResult[T]:
        self._reject_sync_call()
        label: str | None = kwargs.pop("label", None)
        self._apply_kwarg_defaults(kwargs)

        t0 = time.perf_counter()
        result: ParsedCompletionResult[T] | None = None
        error: str | None = None
        try:
            result = self._error_handler.run(
                self._parse_completions, messages, response_format, **kwargs
            )
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=messages,
                response=result.parsed.serialize(by_alias=False) if result else None,
                usage=(result.usage if result else None),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    def _parse_completions[T: BaseModel](
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[T],
        **kwargs,
    ) -> ParsedCompletionResult[T]:
        response: ParsedChatCompletion[T] = self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise LLMIncompleteError(
                "Structured output parsing returned None",
                incomplete_details=None,
            )
        return ParsedCompletionResult[T](parsed=parsed, usage=response.usage)

    async def aparse_completions[T: BaseModel](
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[T],
        **kwargs,
    ) -> ParsedCompletionResult[T]:
        label: str | None = kwargs.pop("label", None)
        self._apply_kwarg_defaults(kwargs)

        t0 = time.perf_counter()
        result: ParsedCompletionResult[T] | None = None
        error: str | None = None
        try:
            result = await self._error_handler.arun(
                self._aparse_completions, messages, response_format, **kwargs
            )
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=messages,
                response=result.parsed.serialize(by_alias=False) if result else None,
                usage=(result.usage if result else None),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    async def _aparse_completions[T: BaseModel](
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[T],
        **kwargs,
    ) -> ParsedCompletionResult[T]:
        response: ParsedChatCompletion[
            T
        ] = await self._async_client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise LLMIncompleteError(
                "Structured output parsing returned None",
                incomplete_details=None,
            )
        return ParsedCompletionResult[T](parsed=parsed, usage=response.usage)

    # Parse — Responses

    def parse_responses[T: BaseModel](
        self,
        input_: ResponseInputParam,
        response_format: type[T],
        **kwargs,
    ) -> ParsedResponseResult[T]:
        self._reject_sync_call()
        label: str | None = kwargs.pop("label", None)
        self._apply_kwarg_defaults(kwargs)

        t0 = time.perf_counter()
        result: ParsedResponseResult[T] | None = None
        error: str | None = None
        try:
            result = self._error_handler.run(
                self._parse_responses, input_, response_format, **kwargs
            )
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=input_,
                response=result.parsed.serialize(by_alias=False) if result else None,
                usage=(result.usage if result else None),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    def _parse_responses[T: BaseModel](
        self,
        input_: ResponseInputParam,
        response_format: type[T],
        **kwargs,
    ) -> ParsedResponseResult[T]:
        response: ParsedResponse[T] = self._client.responses.parse(
            model=self._model,
            input=input_,
            text_format=response_format,
            **kwargs,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise LLMIncompleteError(
                "Structured output parsing returned None",
                incomplete_details=None,
            )
        return ParsedResponseResult[T](parsed=parsed, usage=response.usage)

    async def aparse_responses[T: BaseModel](
        self,
        input_: ResponseInputParam,
        response_format: type[T],
        **kwargs,
    ) -> ParsedResponseResult[T]:
        label: str | None = kwargs.pop("label", None)
        self._apply_kwarg_defaults(kwargs)

        t0 = time.perf_counter()
        result: ParsedResponseResult[T] | None = None
        error: str | None = None
        try:
            result = await self._error_handler.arun(
                self._aparse_responses, input_, response_format, **kwargs
            )
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=input_,
                response=result.parsed.serialize(by_alias=False) if result else None,
                usage=(result.usage if result else None),
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    async def _aparse_responses[T: BaseModel](
        self,
        input_: ResponseInputParam,
        response_format: type[T],
        **kwargs,
    ) -> ParsedResponseResult[T]:
        response: ParsedResponse[T] = await self._async_client.responses.parse(
            model=self._model,
            input=input_,
            text_format=response_format,
            **kwargs,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise LLMIncompleteError(
                "Structured output parsing returned None",
                incomplete_details=None,
            )
        return ParsedResponseResult[T](parsed=parsed, usage=response.usage)

    # Debug capture

    def _record_call(
        self,
        *,
        label: str | None,
        prompt: LLMPrompt,
        response: LLMResponseContent,
        usage: LLMUsage,
        duration_ms: int,
        error: str | None,
    ) -> None:
        """Append a :class:`DebugCall` to the current request's
        :class:`ExecutionContext.debug_trace` AND log a one-liner.

        The trace is only populated when the request is inside a
        pipeline run (signalled by ``ctx.debug_trace.started_at != 0``).
        Outside a pipeline (e.g. standalone LLM calls or admin routes)
        we still log but skip the trace appendage.
        """
        input_tokens, output_tokens, total_tokens = self._extract_token_counts(usage)
        cached_input_tokens, cache_write_tokens = self._extract_cache_tokens(usage)
        effective_label = label or "openai"
        call_id = uuid4()

        UsageTracker().record(
            call_id=call_id,
            model=self._model,
            label=effective_label,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            duration_ms=duration_ms,
            error=error,
        )

        if self._execution_context.debug_trace.started_at:
            capture = cast(SupportsLLMConfig, get_settings()).capture_raw_traces
            self._execution_context.debug_trace.calls.append(
                DebugCall(
                    call_id=call_id,
                    label=effective_label,
                    model=self._model,
                    prompt=list(prompt) if capture else [],
                    response=response if capture else None,
                )
            )

        if error is not None:
            self._logger.error(
                "[llm] %s failed dur=%dms err=%s",
                effective_label,
                duration_ms,
                error,
            )
        else:
            self._logger.info(
                "[llm] %s tokens=%d/%d/%d dur=%dms",
                effective_label,
                input_tokens,
                output_tokens,
                total_tokens,
                duration_ms,
            )

    @staticmethod
    def _extract_token_counts(usage: LLMUsage) -> tuple[int, int, int]:
        """Pull ``(input, output, total)`` token counts out of either
        usage shape. :class:`CompletionUsage` exposes ``prompt_tokens``
        / ``completion_tokens``; :class:`ResponseUsage` exposes
        ``input_tokens`` / ``output_tokens``. We normalise both to
        the ``input/output/total`` triple captured on
        :class:`DebugCall`."""
        if usage is None:
            return 0, 0, 0
        if isinstance(usage, ResponseUsage):
            input_tokens = int(usage.input_tokens or 0)
            output_tokens = int(usage.output_tokens or 0)
            total_tokens = int(usage.total_tokens or input_tokens + output_tokens)
            return input_tokens, output_tokens, total_tokens
        if isinstance(usage, CompletionUsage):
            input_tokens = int(usage.prompt_tokens or 0)
            output_tokens = int(usage.completion_tokens or 0)
            total_tokens = int(usage.total_tokens or input_tokens + output_tokens)
            return input_tokens, output_tokens, total_tokens
        return 0, 0, 0

    @staticmethod
    def _extract_cache_tokens(usage: LLMUsage) -> tuple[int, int]:
        """`(cache reads, cache writes)` from either usage shape.

        Both are subsets of the input count, and billed at their own rates.
        """
        if usage is None:
            return 0, 0
        details = getattr(usage, "input_tokens_details", None) or getattr(
            usage, "prompt_tokens_details", None
        )
        if details is None:
            return 0, 0
        return (
            int(getattr(details, "cached_tokens", 0) or 0),
            int(getattr(details, "cache_write_tokens", 0) or 0),
        )

    # Provider-neutral seam

    def get_service_name(self) -> str:
        return "openai"

    def _apply_generation_knobs(
        self,
        kwargs: dict[str, Any],
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
    ) -> None:
        """Spell the neutral knobs the way the chosen surface wants them."""
        if max_output_tokens is not None:
            key = (
                "max_output_tokens"
                if self._supports_responses_api
                else "max_completion_tokens"
            )
            kwargs.setdefault(key, max_output_tokens)
        if reasoning_effort is not None:
            if self._supports_responses_api:
                kwargs.setdefault("reasoning", {"effort": reasoning_effort})
            else:
                kwargs.setdefault("reasoning_effort", reasoning_effort)

    def _to_llm_result(self, result: ResponseResult | CompletionResult) -> LLMResult:
        """Flatten either wire result into the neutral one."""
        tool_calls = result.tool_calls
        return LLMResult(
            content=result.content,
            tool_calls=tool_calls,
            usage=self._to_usage_counts(result.usage),
            reasoning=(
                result.reasoning if isinstance(result, ResponseResult) else None
            ),
            finish_reason=self._to_finish_reason(result, tool_calls),
        )

    @staticmethod
    def _to_finish_reason(
        result: ResponseResult | CompletionResult,
        tool_calls: list[StoredToolCall] | None,
    ) -> FinishReason:
        """Normalise why generation stopped.

        The Responses API reports a status rather than a reason, and a reply cut
        short has already been raised by `_raise_if_incomplete`, so what reaches
        here either asked for tools or finished on its own.
        """
        if isinstance(result, ResponseResult) or result.finish_reason is None:
            return FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP
        return _FINISH_REASON_MAP.get(result.finish_reason, FinishReason.OTHER)

    def _to_usage_counts(self, usage: LLMUsage) -> LLMUsageCounts | None:
        """Fold either OpenAI usage shape into the neutral counts."""
        if usage is None:
            return None
        input_tokens, output_tokens, _ = self._extract_token_counts(usage)
        cached_input_tokens, cache_write_tokens = self._extract_cache_tokens(usage)
        details = getattr(usage, "output_tokens_details", None) or getattr(
            usage, "completion_tokens_details", None
        )
        return LLMUsageCounts(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
        )

    def _complete(
        self,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        tools: Sequence[Tool[Any]] | None,
        stream_iterator: StreamIterator | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        label: str | None,
        **kwargs,
    ) -> LLMResult:
        self._reject_sync_call()
        stored = list(messages) if messages else None
        self._apply_generation_knobs(kwargs, max_output_tokens, reasoning_effort)

        if self._supports_responses_api:
            if tools:
                kwargs["tools"] = OpenAIMessageAdapter.to_responses_tools(tools)
            result: ResponseResult | CompletionResult = self.responses(
                OpenAIMessageAdapter.to_responses_input(messages=stored, prompt=prompt),
                stream_iterator=stream_iterator,
                label=label,
                **kwargs,
            )
        else:
            if tools:
                kwargs["tools"] = OpenAIMessageAdapter.to_chat_completions_tools(tools)
            result = self.completions(
                OpenAIMessageAdapter.to_chat_completions_messages(
                    messages=stored, prompt=prompt
                ),
                stream_iterator=stream_iterator,
                label=label,
                **kwargs,
            )

        return self._to_llm_result(result)

    async def _acomplete(
        self,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        tools: Sequence[Tool[Any]] | None,
        stream_iterator: StreamIterator | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        label: str | None,
        **kwargs,
    ) -> LLMResult:
        stored = list(messages) if messages else None
        self._apply_generation_knobs(kwargs, max_output_tokens, reasoning_effort)

        if self._supports_responses_api:
            if tools:
                kwargs["tools"] = OpenAIMessageAdapter.to_responses_tools(tools)
            result: ResponseResult | CompletionResult = await self.aresponses(
                OpenAIMessageAdapter.to_responses_input(messages=stored, prompt=prompt),
                stream_iterator=stream_iterator,
                label=label,
                **kwargs,
            )
        else:
            if tools:
                kwargs["tools"] = OpenAIMessageAdapter.to_chat_completions_tools(tools)
            result = await self.acompletions(
                OpenAIMessageAdapter.to_chat_completions_messages(
                    messages=stored, prompt=prompt
                ),
                stream_iterator=stream_iterator,
                label=label,
                **kwargs,
            )

        return self._to_llm_result(result)

    def _parse[T: BaseModel](
        self,
        schema: type[T],
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        label: str | None,
        **kwargs,
    ) -> ParsedResult[T]:
        self._reject_sync_call()
        stored = list(messages) if messages else None
        self._apply_generation_knobs(kwargs, max_output_tokens, reasoning_effort)

        parsed: ParsedResponseResult[T] | ParsedCompletionResult[T]
        if self._supports_responses_api:
            parsed = self.parse_responses(
                OpenAIMessageAdapter.to_responses_input(messages=stored, prompt=prompt),
                schema,
                label=label,
                **kwargs,
            )
        else:
            parsed = self.parse_completions(
                OpenAIMessageAdapter.to_chat_completions_messages(
                    messages=stored, prompt=prompt
                ),
                schema,
                label=label,
                **kwargs,
            )

        return ParsedResult[T](
            parsed=parsed.parsed,
            usage=self._to_usage_counts(parsed.usage),
        )

    async def _aparse[T: BaseModel](
        self,
        schema: type[T],
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        label: str | None,
        **kwargs,
    ) -> ParsedResult[T]:
        stored = list(messages) if messages else None
        self._apply_generation_knobs(kwargs, max_output_tokens, reasoning_effort)

        parsed: ParsedResponseResult[T] | ParsedCompletionResult[T]
        if self._supports_responses_api:
            parsed = await self.aparse_responses(
                OpenAIMessageAdapter.to_responses_input(messages=stored, prompt=prompt),
                schema,
                label=label,
                **kwargs,
            )
        else:
            parsed = await self.aparse_completions(
                OpenAIMessageAdapter.to_chat_completions_messages(
                    messages=stored, prompt=prompt
                ),
                schema,
                label=label,
                **kwargs,
            )

        return ParsedResult[T](
            parsed=parsed.parsed,
            usage=self._to_usage_counts(parsed.usage),
        )
