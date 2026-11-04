"""Talking to a model, without naming one."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, final

from base.models.definitions.model import ReasoningEffort
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredMessage
from base.models.view_models.llm_result import LLMResult, ParsedResult
from base.models.view_models.prompt import RenderedPrompt
from base.services.tool_calling import Tool
from base.tools.stream import StreamIterator


class AbstractLLMService(ABC):
    """One call shape, whatever is behind it.

    Callers hand over `RenderedPrompt` and `StoredMessage`, which belong to
    this library rather than to any provider. Turning those into a wire format
    is the adapter's job, so a caller never learns whose API answered.

    `max_output_tokens` and `reasoning_effort` are stated in neutral terms;
    each adapter spells them the way its provider wants.
    """

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def get_service_name(self) -> str: ...

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
        **kwargs: Any,
    ) -> LLMResult:
        """Complete a turn. Pass a prompt, prior messages, or both.

        `tools` lets the model ask for a call; the reply carries `tool_calls`
        when it does. Running them and asking again is the caller's loop.
        """
        self._require_input(prompt, messages, "complete")
        return self._complete(
            prompt,
            messages,
            tools,
            stream_iterator,
            max_output_tokens,
            reasoning_effort,
            label,
            **kwargs,
        )

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
        **kwargs: Any,
    ) -> LLMResult:
        """Complete a turn. Pass a prompt, prior messages, or both.

        `tools` lets the model ask for a call; the reply carries `tool_calls`
        when it does. Running them and asking again is the caller's loop.
        """
        self._require_input(prompt, messages, "acomplete")
        return await self._acomplete(
            prompt,
            messages,
            tools,
            stream_iterator,
            max_output_tokens,
            reasoning_effort,
            label,
            **kwargs,
        )

    @abstractmethod
    def _parse[T: BaseModel](
        self,
        schema: type[T],
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        max_output_tokens: int | None,
        reasoning_effort: ReasoningEffort | None,
        label: str | None,
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
        **kwargs: Any,
    ) -> ParsedResult[T]:
        """Complete a turn and parse the reply into `schema`.

        Takes no stream iterator: a parsed call returns one object, not deltas.
        """
        self._require_input(prompt, messages, "parse")
        return self._parse(
            schema,
            prompt,
            messages,
            max_output_tokens,
            reasoning_effort,
            label,
            **kwargs,
        )

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
        **kwargs: Any,
    ) -> ParsedResult[T]:
        """Complete a turn and parse the reply into `schema`.

        Takes no stream iterator: a parsed call returns one object, not deltas.
        """
        self._require_input(prompt, messages, "aparse")
        return await self._aparse(
            schema,
            prompt,
            messages,
            max_output_tokens,
            reasoning_effort,
            label,
            **kwargs,
        )

    @staticmethod
    def _require_input(
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
        method: str,
    ) -> None:
        if prompt is None and not messages:
            raise ValueError(f"{method} needs a prompt, messages, or both")
