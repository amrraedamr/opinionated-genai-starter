"""Talking to a model, without naming one."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, final

from base.models.definitions.model import ReasoningEffort
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredMessage
from base.models.view_models.llm_result import LLMResult, ParsedResult
from base.models.view_models.model import CacheHint
from base.models.view_models.prompt import RenderedPrompt
from base.services.rate_limiting.llm_budget import TokenBudget
from base.services.tool_calling import Tool
from base.tools.stream import StreamIterator


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
