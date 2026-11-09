"""Talking to a model, without naming one."""

from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any, final

from base.models.definitions.model import ReasoningEffort
from base.models.definitions.telemetry import (
    GENAI_COMPLETION,
    GENAI_OPERATION,
    GENAI_PROMPT,
    GENAI_REQUEST_MODEL,
    GENAI_RESPONSE_FINISH_REASON,
    GENAI_SYSTEM,
    GENAI_USAGE_CACHE_READ_TOKENS,
    GENAI_USAGE_CACHE_WRITE_TOKENS,
    GENAI_USAGE_INPUT_TOKENS,
    GENAI_USAGE_OUTPUT_TOKENS,
)
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredMessage
from base.models.view_models.llm_result import LLMResult, ParsedResult
from base.models.view_models.model import CacheHint
from base.models.view_models.prompt import RenderedPrompt
from base.services.rate_limiting.llm_budget import TokenBudget
from base.services.tool_calling import Tool
from base.tools.stream import StreamIterator
from base.tools.tracing import OptionalTelemetry


class AbstractLLMService(ABC):
    """One call shape, whatever is behind it.

    Callers hand over `RenderedPrompt` and `StoredMessage`, which belong to
    this library rather than to any provider. Turning those into a wire format
    is the adapter's job, so a caller never learns whose API answered.

    `max_output_tokens` and `reasoning_effort` are stated in neutral terms;
    each adapter spells them the way its provider wants. `cache_hint` is the
    same idea applied to prompt caching, and the one place the providers
    diverge enough to be worth saying out loud: it marks the prompt's head as
    repeated, and each adapter does whatever its provider needs to benefit.
    """

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def get_service_name(self) -> str: ...

    @abstractmethod
    def _apply_cache_hint(
        self,
        request: dict[str, Any],
        cache_hint: CacheHint | None,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None = None,
        tools: Sequence[Tool[Any]] | None = None,
        schema: type[BaseModel] | None = None,
    ) -> None:
        """Ask this provider to cache the prompt's head, however it takes it.

        Every adapter implements this, including the ones with nothing to
        send: prompt caching is the place the providers disagree about
        mechanism rather than spelling, so a provider that cannot act on the
        hint should say so here instead of leaving an argument unread.

        `request` is whatever that adapter hands its SDK, mutated in place.
        The head is passed alongside it because a provider may need to name
        it rather than mark it, and the wire form has no single place where
        instructions, tools and an output schema all appear. `messages` is
        part of that head and not only of the tail: a stored message carrying
        a system or developer role is folded into the instructions by the
        adapters that keep instructions out of the message list, so leaving
        it out would let two calls with different instructions look alike.
        """

    @abstractmethod
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
    ) -> LLMResult: ...

    @abstractmethod
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
    ) -> LLMResult: ...

    @final
    def complete(
        self,
        *,
        prompt: RenderedPrompt | None = None,
        messages: Sequence[StoredMessage] | None = None,
        tools: Sequence[Tool[Any]] | None = None,
        stream_iterator: StreamIterator | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        label: str | None = None,
        cache_hint: CacheHint | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Complete a turn. Pass a prompt, prior messages, or both.

        `tools` lets the model ask for a call; the reply carries `tool_calls`
        when it does. Running them and asking again is the caller's loop.
        """
        self._require_input(prompt, messages, "complete")
        budget = self._budget
        budget.check()
        with self._genai_span("chat", prompt, messages) as span:
            result = self._complete(
                prompt,
                messages,
                tools,
                stream_iterator,
                max_output_tokens,
                reasoning_effort,
                label,
                cache_hint,
                **kwargs,
            )
            self._record_genai(span, result)
        budget.spend(result.usage.total_tokens if result.usage else 0)
        return result

    @final
    async def acomplete(
        self,
        *,
        prompt: RenderedPrompt | None = None,
        messages: Sequence[StoredMessage] | None = None,
        tools: Sequence[Tool[Any]] | None = None,
        stream_iterator: StreamIterator | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        label: str | None = None,
        cache_hint: CacheHint | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Complete a turn. Pass a prompt, prior messages, or both.

        `tools` lets the model ask for a call; the reply carries `tool_calls`
        when it does. Running them and asking again is the caller's loop.
        """
        self._require_input(prompt, messages, "acomplete")
        budget = self._budget
        await budget.acheck()
        with self._genai_span("chat", prompt, messages) as span:
            result = await self._acomplete(
                prompt,
                messages,
                tools,
                stream_iterator,
                max_output_tokens,
                reasoning_effort,
                label,
                cache_hint,
                **kwargs,
            )
            self._record_genai(span, result)
        await budget.aspend(result.usage.total_tokens if result.usage else 0)
        return result

    @abstractmethod
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
    ) -> ParsedResult[T]: ...

    @abstractmethod
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
    ) -> ParsedResult[T]: ...

    @final
    def parse[T: BaseModel](
        self,
        schema: type[T],
        *,
        prompt: RenderedPrompt | None = None,
        messages: Sequence[StoredMessage] | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        label: str | None = None,
        cache_hint: CacheHint | None = None,
        **kwargs: Any,
    ) -> ParsedResult[T]:
        """Complete a turn and parse the reply into `schema`.

        Takes no stream iterator: a parsed call returns one object, not deltas.
        """
        self._require_input(prompt, messages, "parse")
        budget = self._budget
        budget.check()
        result = self._parse(
            schema,
            prompt,
            messages,
            max_output_tokens,
            reasoning_effort,
            label,
            cache_hint,
            **kwargs,
        )
        budget.spend(result.usage.total_tokens if result.usage else 0)
        return result

    @final
    async def aparse[T: BaseModel](
        self,
        schema: type[T],
        *,
        prompt: RenderedPrompt | None = None,
        messages: Sequence[StoredMessage] | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        label: str | None = None,
        cache_hint: CacheHint | None = None,
        **kwargs: Any,
    ) -> ParsedResult[T]:
        """Complete a turn and parse the reply into `schema`.

        Takes no stream iterator: a parsed call returns one object, not deltas.
        """
        self._require_input(prompt, messages, "aparse")
        budget = self._budget
        await budget.acheck()
        result = await self._aparse(
            schema,
            prompt,
            messages,
            max_output_tokens,
            reasoning_effort,
            label,
            cache_hint,
            **kwargs,
        )
        await budget.aspend(result.usage.total_tokens if result.usage else 0)
        return result

    @contextmanager
    def _genai_span(
        self,
        operation: str,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
    ) -> Generator[Any]:
        """A span describing the call, following the GenAI conventions.

        Opened here rather than in each adapter: this is the boundary every
        provider crosses, so one span covers all of them and none of them has
        to remember to emit it.

        Prompt content is an attribute only where a service asked for it.
        A trace is routinely shipped to a third party, and the prompt is the
        one thing in it that belongs to somebody else.
        """
        attributes: dict[str, Any] = {
            GENAI_SYSTEM: self.get_service_name(),
            GENAI_OPERATION: operation,
            GENAI_REQUEST_MODEL: self.model,
        }
        if self._capture_content():
            attributes[GENAI_PROMPT] = self._content_of(prompt, messages)

        with OptionalTelemetry.continued_span(
            f"{operation} {self.model}", None, kind="client", attributes=attributes
        ) as span:
            yield span

    def _record_genai(self, span: Any, result: Any) -> None:
        """Put what the call cost on the span, once the provider has answered."""
        if span is None:
            return
        usage = getattr(result, "usage", None)
        if usage is not None:
            span.set_attribute(GENAI_USAGE_INPUT_TOKENS, usage.input_tokens)
            span.set_attribute(GENAI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
            span.set_attribute(GENAI_USAGE_CACHE_READ_TOKENS, usage.cached_input_tokens)
            span.set_attribute(GENAI_USAGE_CACHE_WRITE_TOKENS, usage.cache_write_tokens)
        finish = getattr(result, "finish_reason", None)
        if finish is not None:
            span.set_attribute(GENAI_RESPONSE_FINISH_REASON, str(finish.value))
        content = getattr(result, "content", None)
        if content and self._capture_content():
            span.set_attribute(GENAI_COMPLETION, content)

    @staticmethod
    def _capture_content() -> bool:
        """Whether this service opted into prompts and replies in its traces.

        Read from telemetry rather than settings: it already holds the answer,
        and a service without the `otel` extra has no spans to put them on.
        """
        return OptionalTelemetry.capture_content()

    @staticmethod
    def _content_of(
        prompt: RenderedPrompt | None, messages: Sequence[StoredMessage] | None
    ) -> str:
        """The text going to the model, for a service that asked to record it."""
        parts: list[str] = []
        if prompt is not None:
            parts.extend(instruction.content for instruction in prompt.instructions)
            parts.extend(message.content for message in prompt.messages)
        if messages:
            parts.extend(message.content or "" for message in messages)
        return "\n".join(part for part in parts if part)

    @property
    def _budget(self) -> TokenBudget:
        """Resolved on first use, because a service may be built before settings.

        Held per service instance rather than per call: it is stateless, and
        the limiter behind it is the shared one.
        """
        existing = getattr(self, "_token_budget", None)
        if existing is None:
            existing = TokenBudget()
            self._token_budget = existing
        return existing

    @staticmethod
    def _require_input(
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        method: str,
    ) -> None:
        if prompt is None and not messages:
            raise ValueError(f"{method} needs a prompt, messages, or both")
