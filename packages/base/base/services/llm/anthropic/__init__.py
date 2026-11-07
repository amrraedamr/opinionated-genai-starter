"""Anthropic behind the neutral LLM seam."""

import json
import time
from collections.abc import Sequence
from typing import Any, cast
from uuid import uuid4

from base.tools.errors import MissingExtraError

try:
    import httpx2
    from anthropic import Anthropic, AnthropicError, AsyncAnthropic
    from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "anthropic", "anthropic", feature="AnthropicService"
    ) from exc

from base.clients.credentials import OutboundAuth
from base.models.definitions.model import (
    CachePersistence,
    FinishReason,
    ProviderType,
    ReasoningEffort,
)
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import (
    StoredMessage,
    StoredToolCall,
    StoredToolCallFunction,
)
from base.models.view_models.debug_trace import DebugCall, LLMResponseContent
from base.models.view_models.exceptions.llm import (
    LLMConfigError,
    LLMContentFilterError,
    LLMError,
    LLMStreamError,
)
from base.models.view_models.llm_result import LLMResult, LLMUsageCounts, ParsedResult
from base.models.view_models.model import CacheHint, ModelConfigurations
from base.models.view_models.prompt import RenderedPrompt
from base.models.view_models.streams import (
    StreamEnd,
    StreamError,
    StreamMessage,
    StreamStart,
)
from base.services.common.meta import KeyedSingletonABC
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsLLMConfig
from base.services.error_handling.llm.anthropic import AnthropicErrorHandler
from base.services.execution_context import ExecutionContext
from base.services.llm import AbstractLLMService
from base.services.llm.anthropic.adapter import AnthropicMessageAdapter
from base.services.logging.console import ConsoleLoggerService
from base.services.tool_calling import Tool
from base.services.usage import UsageTracker
from base.tools.stream import StreamIterator

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_CALLS,
    "max_tokens": FinishReason.LENGTH,
    "refusal": FinishReason.CONTENT_FILTER,
}

_CACHE_TTL_MAP: dict[CachePersistence, str] = {
    CachePersistence.SHORT: "5m",
    CachePersistence.LONG: "1h",
}
"""Anthropic spells the two persistence levels as durations."""


class AnthropicService(AbstractLLMService, metaclass=KeyedSingletonABC):
    """Anthropic's Messages API, speaking the neutral seam."""

    DEFAULT_MAX_OUTPUT_TOKENS: int = 4096
    """Anthropic requires `max_tokens`, so a call that omits it gets this."""

    def __init__(
        self,
        config: ModelConfigurations | None = None,
        auth: OutboundAuth | None = None,
        verify: bool = True,
    ) -> None:
        if not config:
            config = ModelConfigurations.get_default_for_provider(
                ProviderType.ANTHROPIC
            )
        if config.model_name is None:
            raise LLMConfigError("model_name is required")

        self._config: ModelConfigurations = config
        self._model: str = config.model_name
        self._auth: OutboundAuth | None = auth
        self._verify: bool = verify
        self._logger: ConsoleLoggerService = ConsoleLoggerService()
        self._client, self._async_client = self._initialize_clients()
        self._error_handler: AnthropicErrorHandler = AnthropicErrorHandler(
            retries=config.api_retries,
            base_wait=config.wait_time,
        )

    def _initialize_clients(self) -> tuple[Anthropic, AsyncAnthropic]:
        """Both surfaces share one configuration.

        SDK-level retries are off: `AnthropicErrorHandler` owns retry policy,
        and two layers of backoff would multiply the wait.

        A credential or a custom TLS setting needs its own transport, so those
        cases build `httpx2` clients; otherwise the SDK makes its own.
        """
        settings: dict[str, Any] = {
            "api_key": self._config.api_key,
            "base_url": self._config.api_host or None,
            "max_retries": 0,
        }
        if self._auth is None and self._verify:
            return Anthropic(**settings), AsyncAnthropic(**settings)

        return (
            Anthropic(
                **settings,
                http_client=httpx2.Client(auth=self._auth, verify=self._verify),
            ),
            AsyncAnthropic(
                **settings,
                http_client=httpx2.AsyncClient(auth=self._auth, verify=self._verify),
            ),
        )

    def _reject_sync_call(self) -> None:
        """Refuse a sync call when credentials are acquired asynchronously.

        Letting it reach the SDK surfaces as a connection error, which the
        retry policy treats as transient — so a permanent misconfiguration
        would be retried and then reported as a connection failure.
        """
        if self._auth is not None:
            raise LLMConfigError(
                "This AnthropicService authenticates with an asynchronous "
                "credential and has no synchronous surface. Use the "
                "`a`-prefixed methods."
            )

    @property
    def model(self) -> str:
        return self._model

    def get_service_name(self) -> str:
        return "anthropic"

    @property
    def _execution_context(self) -> ExecutionContext:
        """Resolved per access: the service is a singleton, the context is not."""
        return ExecutionContext()

    def _complete(
        self,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        tools: Sequence[Tool[Any]] | None,
        stream_iterator: StreamIterator | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        label: str | None,
        cache_hint: CacheHint | None,
        **kwargs: Any,
    ) -> LLMResult:
        self._reject_sync_call()
        wire, request = self._prepare_completion(
            prompt,
            messages,
            tools,
            max_output_tokens,
            reasoning_effort,
            cache_hint,
            **kwargs,
        )

        t0 = time.perf_counter()
        result: LLMResult | None = None
        error: str | None = None
        try:
            if stream_iterator is not None:
                result = self._stream(request, stream_iterator)
            else:
                message = self._error_handler.run(
                    self._client.messages.create, **request
                )
                result = self._to_result(cast(Message, message))
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=wire,
                response=result.content if result else None,
                usage=result.usage if result else None,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    async def _acomplete(
        self,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        tools: Sequence[Tool[Any]] | None,
        stream_iterator: StreamIterator | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        label: str | None,
        cache_hint: CacheHint | None,
        **kwargs: Any,
    ) -> LLMResult:
        wire, request = self._prepare_completion(
            prompt,
            messages,
            tools,
            max_output_tokens,
            reasoning_effort,
            cache_hint,
            **kwargs,
        )

        t0 = time.perf_counter()
        result: LLMResult | None = None
        error: str | None = None
        try:
            if stream_iterator is not None:
                result = await self._astream(request, stream_iterator)
            else:
                message = await self._error_handler.arun(
                    self._async_client.messages.create, **request
                )
                result = self._to_result(cast(Message, message))
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=wire,
                response=result.content if result else None,
                usage=result.usage if result else None,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    def _parse[T: BaseModel](
        self,
        schema: type[T],
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        label: str | None,
        cache_hint: CacheHint | None,
        **kwargs: Any,
    ) -> ParsedResult[T]:
        """Anthropic has no structured-output mode, so this forces a tool call.

        Offering one tool whose schema is the target and requiring it be called
        is the documented way to get a guaranteed shape back.
        """
        self._reject_sync_call()
        wire, request = self._prepare_parse(
            schema,
            prompt,
            messages,
            max_output_tokens,
            reasoning_effort,
            cache_hint,
            **kwargs,
        )

        t0 = time.perf_counter()
        result: ParsedResult[T] | None = None
        error: str | None = None
        try:
            message = self._error_handler.run(self._client.messages.create, **request)
            result = self._to_parsed_result(schema, cast(Message, message))
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=wire,
                response=result.parsed.serialize(by_alias=False) if result else None,
                usage=result.usage if result else None,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    async def _aparse[T: BaseModel](
        self,
        schema: type[T],
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        label: str | None,
        cache_hint: CacheHint | None,
        **kwargs: Any,
    ) -> ParsedResult[T]:
        """Anthropic has no structured-output mode, so this forces a tool call.

        Offering one tool whose schema is the target and requiring it be called
        is the documented way to get a guaranteed shape back.
        """
        wire, request = self._prepare_parse(
            schema,
            prompt,
            messages,
            max_output_tokens,
            reasoning_effort,
            cache_hint,
            **kwargs,
        )

        t0 = time.perf_counter()
        result: ParsedResult[T] | None = None
        error: str | None = None
        try:
            message = await self._error_handler.arun(
                self._async_client.messages.create, **request
            )
            result = self._to_parsed_result(schema, cast(Message, message))
            return result
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=wire,
                response=result.parsed.serialize(by_alias=False) if result else None,
                usage=result.usage if result else None,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    def _prepare_completion(
        self,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        tools: Sequence[Tool[Any]] | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        cache_hint: CacheHint | None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        wire, system = AnthropicMessageAdapter.to_messages(
            messages=messages, prompt=prompt
        )
        request = self._build_request(
            wire, system, max_output_tokens, reasoning_effort, **kwargs
        )
        if tools:
            request["tools"] = AnthropicMessageAdapter.to_tools(tools)
        self._apply_cache_hint(
            request, cache_hint, prompt, messages=messages, tools=tools
        )
        return wire, request

    def _prepare_parse[T: BaseModel](
        self,
        schema: type[T],
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        cache_hint: CacheHint | None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        wire, system = AnthropicMessageAdapter.to_messages(
            messages=messages, prompt=prompt
        )
        request = self._build_request(
            wire, system, max_output_tokens, reasoning_effort, **kwargs
        )
        request["output_config"] = dict(request.get("output_config") or {})
        request["output_config"]["format"] = {
            "type": "json_schema",
            "schema": schema.model_json_schema(),
        }
        self._apply_cache_hint(
            request, cache_hint, prompt, messages=messages, schema=schema
        )
        return wire, request

    def _to_parsed_result[T: BaseModel](
        self, schema: type[T], message: Message
    ) -> ParsedResult[T]:
        """Read the schema-constrained reply, which arrives as ordinary text.

        Asking for a JSON output format constrains decoding, so the answer
        comes back in a text block rather than as a tool call. Thinking, when
        the model did any, arrives in its own block and is skipped here.
        """
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                return ParsedResult[T](
                    parsed=schema.model_validate_json(block.text),
                    usage=self._to_usage_counts(message),
                )
        raise LLMError("Anthropic returned no structured output")

    def _apply_cache_hint(
        self,
        request: dict[str, Any],
        cache_hint: CacheHint | None,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None = None,
        tools: Sequence[Tool[Any]] | None = None,
        schema: type[BaseModel] | None = None,
    ) -> None:
        """Mark where the cached prefix ends, which is all Anthropic needs.

        Anthropic caches nothing unless a request says where to stop, and the
        prefix runs tools first and then the system block. One mark on the
        system text therefore covers both, and the conversation after it stays
        outside. With no instructions to mark, the tools are the whole of what
        can be cached and the last of them carries it instead.

        The mark is added to the text rather than replacing it: a hinted call
        has to match the unhinted ones before it, which it cannot do if asking
        to cache rewrites the thing being cached.
        """
        if cache_hint is None:
            return
        control = {"type": "ephemeral", "ttl": _CACHE_TTL_MAP[cache_hint.persistence]}
        system = request.get("system")
        if isinstance(system, str):
            request["system"] = [
                {"type": "text", "text": system, "cache_control": control}
            ]
        elif request.get("tools"):
            request["tools"][-1]["cache_control"] = control

    def _build_request(
        self,
        wire: list[dict[str, Any]],
        system: str | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": wire,
            "max_tokens": max_output_tokens or self.DEFAULT_MAX_OUTPUT_TOKENS,
            **kwargs,
        }
        if system:
            request["system"] = system
        if (
            reasoning_effort is not None
            and "thinking" not in request
            and "effort" not in request.get("output_config", {})
        ):
            # Anthropic now takes an effort word where it used to take a token
            # budget, and the newer models reject the budget outright. The two
            # vocabularies happen to agree, so the neutral value goes straight
            # across; a caller wanting the old manual control can still pass
            # `thinking` through kwargs, which is left alone here.
            if reasoning_effort is ReasoningEffort.NONE:
                request["thinking"] = {"type": "disabled"}
            else:
                request.setdefault("output_config", {})["effort"] = (
                    reasoning_effort.value
                )
        return request

    def _stream(
        self, request: dict[str, Any], stream_iterator: StreamIterator
    ) -> LLMResult:
        """Stream text to the caller and return the assembled message.

        Emits the same `StreamStart` / `StreamMessage` / `StreamEnd` sequence
        the OpenAI adapter does, so a consumer cannot tell the two apart.
        """
        try:
            with self._client.messages.stream(**request) as stream:
                stream_iterator.send(StreamStart())
                for text in stream.text_stream:
                    stream_iterator.send(StreamMessage(data=text))
                message = stream.get_final_message()
        except AnthropicError as exc:
            self._logger.error(f"Error during streaming response: {exc}")
            stream_iterator.send(self._build_stream_error(exc))
            raise LLMStreamError(f"Stream failed: {exc}", provider_error=exc) from exc

        stream_iterator.send(StreamEnd())
        return self._to_result(message)

    async def _astream(
        self, request: dict[str, Any], stream_iterator: StreamIterator
    ) -> LLMResult:
        """Stream text to the caller and return the assembled message.

        Emits the same `StreamStart` / `StreamMessage` / `StreamEnd` sequence
        the OpenAI adapter does, so a consumer cannot tell the two apart.
        """
        try:
            async with self._client.messages.stream(**request) as stream:
                await stream_iterator.asend(StreamStart())
                async for text in stream.text_stream:
                    await stream_iterator.asend(StreamMessage(data=text))
                message = await stream.get_final_message()
        except AnthropicError as exc:
            self._logger.error(f"Error during streaming response: {exc}")
            await stream_iterator.asend(self._build_stream_error(exc))
            raise LLMStreamError(f"Stream failed: {exc}", provider_error=exc) from exc

        await stream_iterator.asend(StreamEnd())
        return self._to_result(message)

    @staticmethod
    def _build_stream_error(error: Exception) -> StreamError:
        return StreamError(
            error_code=500,
            text=str(error),
            detail={
                "exception_type": type(error).__name__,
                "exception_module": type(error).__module__,
            },
        )

    def _to_result(self, message: Message) -> LLMResult:
        if message.stop_reason == "refusal":
            raise LLMContentFilterError(
                "Response refused by the model", incomplete_details=None
            )

        text: list[str] = []
        thinking: list[str] = []
        tool_calls: list[StoredToolCall] = []

        for block in message.content:
            if isinstance(block, TextBlock):
                text.append(block.text)
            elif isinstance(block, ThinkingBlock):
                thinking.append(block.thinking)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    StoredToolCall(
                        id=block.id,
                        function=StoredToolCallFunction(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )

        return LLMResult(
            content="".join(text) or None,
            tool_calls=tool_calls or None,
            usage=self._to_usage_counts(message),
            reasoning="".join(thinking) or None,
            finish_reason=_FINISH_REASON_MAP.get(
                message.stop_reason or "", FinishReason.OTHER
            ),
        )

    @staticmethod
    def _to_usage_counts(message: Message) -> LLMUsageCounts:
        """Normalise Anthropic's additive cache counts into the neutral shape.

        Anthropic reports cache reads and writes alongside `input_tokens`
        rather than inside it, while `LLMUsageCounts` treats both as a share of
        the input total. Folding them in here keeps cost maths comparable
        across providers.
        """
        usage = message.usage
        cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        written = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        details = getattr(usage, "output_tokens_details", None)
        return LLMUsageCounts(
            input_tokens=int(usage.input_tokens or 0) + cached + written,
            output_tokens=int(usage.output_tokens or 0),
            cached_input_tokens=cached,
            cache_write_tokens=written,
            reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
        )

    def _record_call(
        self,
        *,
        label: str | None,
        prompt: list[dict[str, Any]],
        response: LLMResponseContent,
        usage: LLMUsageCounts | None,
        duration_ms: int,
        error: str | None,
    ) -> None:
        """Record the call against usage, and against the trace when one runs."""
        effective_label = label or "anthropic"
        call_id = uuid4()
        counts = usage or LLMUsageCounts()

        UsageTracker().record(
            call_id=call_id,
            model=self._model,
            label=effective_label,
            input_tokens=counts.input_tokens,
            output_tokens=counts.output_tokens,
            cached_input_tokens=counts.cached_input_tokens,
            cache_write_tokens=counts.cache_write_tokens,
            duration_ms=duration_ms,
            error=error,
        )

        if self._execution_context.debug_trace.started_at:
            capture = settings_for(SupportsLLMConfig).capture_raw_traces
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
                "[llm] %s failed dur=%dms err=%s", effective_label, duration_ms, error
            )
        else:
            self._logger.info(
                "[llm] %s tokens=%d/%d/%d dur=%dms",
                effective_label,
                counts.input_tokens,
                counts.output_tokens,
                counts.total_tokens,
                duration_ms,
            )
