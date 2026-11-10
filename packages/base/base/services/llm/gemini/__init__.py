"""Google Gemini behind the neutral LLM seam."""

import json
import time
from base64 import b64encode
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from base.tools.errors import MissingExtraError

try:
    from google import genai
    from google.genai import types as genai_types
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("gemini", "google-genai", feature="GeminiService") from exc

from base.models.definitions.model import FinishReason, ProviderType, ReasoningEffort
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
from base.services.configuration.protocols import (
    SupportsGeminiConfig,
    SupportsLLMConfig,
)
from base.services.error_handling.llm.gemini import GeminiErrorHandler
from base.services.execution_context import ExecutionContext
from base.services.llm import AbstractLLMService
from base.services.llm.gemini.adapter import GeminiContentAdapter
from base.services.logging.console import ConsoleLoggerService
from base.services.tool_calling import Tool
from base.services.usage import UsageTracker
from base.tools.stream import StreamIterator

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,
    "IMAGE_SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "MALFORMED_FUNCTION_CALL": FinishReason.OTHER,
}

_BLOCKED_REASONS: frozenset[str] = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "RECITATION"}
)


class GeminiService(AbstractLLMService, metaclass=KeyedSingletonABC):
    """Gemini's generateContent API, speaking the neutral seam."""

    DEFAULT_THINKING_BUDGETS: dict[ReasoningEffort, int] = {
        ReasoningEffort.LOW: 1024,
        ReasoningEffort.MEDIUM: 4096,
        ReasoningEffort.HIGH: 16384,
        ReasoningEffort.XHIGH: 24576,
        ReasoningEffort.MAX: 32768,
    }
    """Gemini budgets thinking in tokens where others take an effort label."""

    def __init__(self, config: ModelConfigurations | None = None) -> None:
        if not config:
            config = ModelConfigurations.get_default_for_provider(ProviderType.GEMINI)
        if config.model_name is None:
            raise LLMConfigError("model_name is required")

        self._config: ModelConfigurations = config
        self._model: str = config.model_name
        self._logger: ConsoleLoggerService = ConsoleLoggerService()
        self._client: genai.Client = self._initialize_client()
        self._error_handler: GeminiErrorHandler = GeminiErrorHandler(
            retries=config.api_retries,
            base_wait=config.wait_time,
        )

    def _initialize_client(self) -> genai.Client:
        """One client serves both surfaces; the async one hangs off `.aio`.

        Vertex AI reads credentials from the environment and needs a project
        and a location, while the Developer API takes an API key, so the two
        are configured differently rather than by the same fields.
        """
        settings = settings_for(SupportsGeminiConfig)
        http_options = (
            genai_types.HttpOptions(base_url=self._config.api_host)
            if self._config.api_host
            else None
        )
        if getattr(settings, "gemini_use_vertex", False):
            return genai.Client(
                vertexai=True,
                project=settings.gemini_project or None,
                location=settings.gemini_location or None,
                http_options=http_options,
            )
        return genai.Client(api_key=self._config.api_key, http_options=http_options)

    @property
    def model(self) -> str:
        return self._model

    def get_service_name(self) -> str:
        return "gemini"

    @property
    def _execution_context(self) -> ExecutionContext:
        """Resolved per access: the service is a singleton, the context is not."""
        return ExecutionContext()

    # Completion

    def _complete(
        self,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        tools: Sequence[Tool[BaseModel]] | None,
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
                result = self._stream(request, stream_iterator)
            else:
                response = self._error_handler.run(
                    self._client.models.generate_content, **request
                )
                result = self._to_result(response)
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
        tools: Sequence[Tool[BaseModel]] | None,
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
                response = await self._error_handler.arun(
                    self._client.aio.models.generate_content, **request
                )
                result = self._to_result(response)
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

    # Structured output

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
        parsed: ParsedResult[T] | None = None
        error: str | None = None
        try:
            response = self._error_handler.run(
                self._client.models.generate_content, **request
            )
            parsed = self._to_parsed_result(schema, response)
            return parsed
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=wire,
                response=parsed.parsed.serialize() if parsed else None,
                usage=parsed.usage if parsed else None,
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
        parsed: ParsedResult[T] | None = None
        error: str | None = None
        try:
            response = await self._error_handler.arun(
                self._client.aio.models.generate_content, **request
            )
            parsed = self._to_parsed_result(schema, response)
            return parsed
        except LLMError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_call(
                label=label,
                prompt=wire,
                response=parsed.parsed.serialize() if parsed else None,
                usage=parsed.usage if parsed else None,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=error,
            )

    # Requests

    def _prepare_completion(
        self,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        tools: Sequence[Tool[BaseModel]] | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        cache_hint: CacheHint | None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        wire, system = GeminiContentAdapter.to_contents(
            messages=messages, prompt=prompt
        )
        config = self._build_config(
            system, max_output_tokens, reasoning_effort, **kwargs
        )
        if tools:
            config["tools"] = GeminiContentAdapter.to_tools(tools)
        self._apply_cache_hint(
            config, cache_hint, prompt, messages=messages, tools=tools
        )
        return wire, {"model": self._model, "contents": wire, "config": config}

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
        """Structured output is a response schema rather than a forced tool.

        Gemini constrains decoding to the schema, so unlike the Anthropic
        adapter there is no tool call to unwrap and no refusal path to detect.
        """
        wire, system = GeminiContentAdapter.to_contents(
            messages=messages, prompt=prompt
        )
        config = self._build_config(
            system, max_output_tokens, reasoning_effort, **kwargs
        )
        config["response_mime_type"] = "application/json"
        config["response_schema"] = GeminiContentAdapter.to_schema(
            schema.model_json_schema()
        )
        self._apply_cache_hint(
            config, cache_hint, prompt, messages=messages, schema=schema
        )
        return wire, {"model": self._model, "contents": wire, "config": config}

    def _apply_cache_hint(
        self,
        request: dict[str, Any],
        cache_hint: CacheHint | None,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None = None,
        tools: Sequence[Tool[BaseModel]] | None = None,
        schema: type[BaseModel] | None = None,
    ) -> None:
        """Nothing to send: Gemini has no per-request cache control.

        It caches a repeated prefix by itself on 2.5 and later, so there is
        nothing to switch on, and it offers no routing key to raise the odds
        with. Its other caching is a `CachedContent` resource with a create,
        a TTL, a storage charge and a delete, which needs an owner that can
        decide when it dies rather than a hint attached to one call.

        A caller holding such a resource already has a way through, since
        `**kwargs` reaches the config unaltered:

            await service.acomplete(
                prompt=prompt, cached_content="cachedContents/abc123"
            )

        Either way a cache read is reported as `cached_input_tokens`, so the
        saving is visible whether or not anything was asked for.
        """

    def _build_config(
        self,
        system: str | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config: dict[str, Any] = dict(kwargs)
        if system:
            config["system_instruction"] = system
        if max_output_tokens is not None:
            config["max_output_tokens"] = max_output_tokens
        if (
            reasoning_effort is not None
            and reasoning_effort is not ReasoningEffort.NONE
            and "thinking_config" not in config
        ):
            budget = self.DEFAULT_THINKING_BUDGETS.get(reasoning_effort)
            if budget is not None:
                config["thinking_config"] = {
                    "thinking_budget": budget,
                    "include_thoughts": True,
                }
        return config

    # Streaming

    def _stream(
        self, request: dict[str, Any], stream_iterator: StreamIterator
    ) -> LLMResult:
        stream_iterator.send(StreamStart())
        chunks: list[Any] = []
        try:
            stream = self._error_handler.run(
                self._client.models.generate_content_stream, **request
            )
            for chunk in stream:
                chunks.append(chunk)
                text = getattr(chunk, "text", None)
                if text:
                    stream_iterator.send(StreamMessage(data=text))
        except Exception as exc:
            stream_iterator.send(self._build_stream_error(exc))
            raise
        finally:
            stream_iterator.send(StreamEnd())
        return self._to_result_from_chunks(chunks)

    async def _astream(
        self, request: dict[str, Any], stream_iterator: StreamIterator
    ) -> LLMResult:
        await stream_iterator.asend(StreamStart())
        chunks: list[Any] = []
        try:
            stream = await self._error_handler.arun(
                self._client.aio.models.generate_content_stream, **request
            )
            async for chunk in stream:
                chunks.append(chunk)
                text = getattr(chunk, "text", None)
                if text:
                    await stream_iterator.asend(StreamMessage(data=text))
        except Exception as exc:
            await stream_iterator.asend(self._build_stream_error(exc))
            raise
        finally:
            await stream_iterator.asend(StreamEnd())
        return self._to_result_from_chunks(chunks)

    @staticmethod
    def _build_stream_error(error: Exception) -> StreamError:
        return StreamError(
            text=str(error),
            detail={
                "exception_type": type(error).__name__,
                "exception_module": type(error).__module__,
            },
        )

    # Results

    def _to_result(self, response: Any) -> LLMResult:
        candidate = self._first_candidate(response)
        finish = self._finish_name(candidate)
        if finish in _BLOCKED_REASONS:
            raise LLMContentFilterError(
                f"Response blocked by the model ({finish})", incomplete_details=None
            )

        text: list[str] = []
        thinking: list[str] = []
        tool_calls: list[StoredToolCall] = []
        for part in self._parts(candidate):
            if getattr(part, "function_call", None) is not None:
                call = part.function_call
                tool_calls.append(
                    StoredToolCall(
                        id=getattr(call, "id", None) or str(uuid4()),
                        function=StoredToolCallFunction(
                            name=call.name or "",
                            arguments=json.dumps(dict(call.args or {})),
                        ),
                        provider_signature=self._signature_of(part),
                    )
                )
            elif getattr(part, "text", None):
                if getattr(part, "thought", False):
                    thinking.append(part.text)
                else:
                    text.append(part.text)

        return LLMResult(
            content="".join(text) or None,
            tool_calls=tool_calls or None,
            usage=self._to_usage_counts(response),
            reasoning="".join(thinking) or None,
            finish_reason=_FINISH_REASON_MAP.get(finish, FinishReason.OTHER),
        )

    def _to_result_from_chunks(self, chunks: list[Any]) -> LLMResult:
        """A streamed call reported piecewise, as one result.

        Usage arrives on the last chunk that carries it rather than on every
        one, so the final occurrence wins.
        """
        if not chunks:
            raise LLMStreamError("Gemini stream produced no chunks")
        last_with_usage = next(
            (c for c in reversed(chunks) if getattr(c, "usage_metadata", None)),
            chunks[-1],
        )
        merged = self._to_result(chunks[-1])
        text = "".join(
            getattr(chunk, "text", "") or ""
            for chunk in chunks
            if getattr(chunk, "text", None)
        )
        return LLMResult(
            content=text or merged.content,
            tool_calls=merged.tool_calls,
            usage=self._to_usage_counts(last_with_usage),
            reasoning=merged.reasoning,
            finish_reason=merged.finish_reason,
        )

    def _to_parsed_result[T: BaseModel](
        self, schema: type[T], response: Any
    ) -> ParsedResult[T]:
        text = getattr(response, "text", None)
        if not text:
            raise LLMError("Gemini returned no structured output")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("Gemini returned output that is not valid JSON") from exc
        return ParsedResult[T](
            parsed=schema.model_validate(payload),
            usage=self._to_usage_counts(response),
        )

    @staticmethod
    def _first_candidate(response: Any) -> Any:
        candidates = getattr(response, "candidates", None) or []
        return candidates[0] if candidates else None

    @staticmethod
    def _finish_name(candidate: Any) -> str:
        reason = getattr(candidate, "finish_reason", None)
        if reason is None:
            return ""
        return getattr(reason, "name", None) or str(reason)

    @staticmethod
    def _signature_of(part: Any) -> str | None:
        """The thought signature this part carries, as text."""
        raw = getattr(part, "thought_signature", None)
        if not raw:
            return None
        return b64encode(raw).decode() if isinstance(raw, bytes) else str(raw)

    @staticmethod
    def _parts(candidate: Any) -> list[Any]:
        content = getattr(candidate, "content", None)
        return list(getattr(content, "parts", None) or [])

    @staticmethod
    def _to_usage_counts(response: Any) -> LLMUsageCounts:
        """Normalise Gemini's separate thinking count into the neutral shape.

        Gemini reports `thoughts_token_count` alongside `candidates_token_count`
        rather than inside it, while `LLMUsageCounts` treats reasoning as a
        share of the output total — so the two are added. Cached input needs no
        such handling: `prompt_token_count` already includes it, which is the
        arrangement the neutral shape expects.
        """
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return LLMUsageCounts()
        thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
        return LLMUsageCounts(
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0)
            + thoughts,
            cached_input_tokens=int(
                getattr(usage, "cached_content_token_count", 0) or 0
            ),
            reasoning_tokens=thoughts,
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
        effective_label = label or "gemini"
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
