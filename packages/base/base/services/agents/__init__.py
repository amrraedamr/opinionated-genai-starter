from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from base.models.definitions.canned_messages import CannedMessages
from base.models.definitions.guardrails import (
    BLOCKED_HISTORY_PLACEHOLDER,
    GuardrailStage,
)
from base.models.definitions.message import MessageRole
from base.models.definitions.model import ReasoningEffort
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import ChatHistoryConfig, StoredMessage
from base.models.view_models.exceptions.llm import LLMError
from base.models.view_models.flow import FlowAbortError
from base.models.view_models.llm_result import LLMResult, ParsedResult
from base.models.view_models.model import CacheHint, ModelConfigurations
from base.models.view_models.prompt import RenderedPrompt
from base.models.view_models.streams import StreamEnd, StreamMessage
from base.services.chat_history import ChatHistoryService
from base.services.common.meta import SingletonABC
from base.services.execution_context import ExecutionContext
from base.services.guardrails import GuardrailBlocked
from base.services.guardrails.factory import (
    build_guarded_stream,
    build_guardrail_pipeline,
)
from base.services.guardrails.pipeline import GuardrailPipeline
from base.services.guardrails.streaming import GuardedStreamIterator
from base.services.llm import AbstractLLMService
from base.services.llm.openai import OpenAIService
from base.services.logging.console import ConsoleLoggerService
from base.services.prompt.service import PromptService
from base.services.store.chat_history import ChatHistoryStore
from base.services.store.chat_history.memory import InMemoryChatHistoryStore
from base.services.tool_calling import Tool
from base.tools.stream import StreamIterator


class AbstractAgent(metaclass=SingletonABC):
    """Template-method base for every LLM agent in a chat / pipeline turn.

    All three hooks are async and take no arguments — agents read from
    and write to the shared :class:`ExecutionContext.flow_variables`.

    On a fatal LLM error, ``arun`` maps the exception to a
    :class:`CannedMessages`, streams it inline (when the request is
    streaming), and raises :class:`FlowAbortError` for the pipeline to
    catch.

    Per-call debug capture lives inside the LLM adapter, around every call it
    makes. Concrete agents pass ``label=...`` to the LLM service so the
    captured :class:`DebugCall` carries a meaningful identifier — by
    convention ``label=type(self).__name__`` for single-shot agents and
    ``label=f"{type(self).__name__}:{item_id}"`` when fanout calls need
    a stable per-item discriminator.
    """

    PROMPT_KEY: ClassVar[str] = ""
    MAX_OUTPUT_TOKENS: ClassVar[int] = 1000
    REASONING_EFFORT: ClassVar[ReasoningEffort | None] = None
    CACHE_HINT: ClassVar[CacheHint | None] = None
    """Set it where an agent's instructions and tools are long enough to be
    worth caching; an agent repeats both on every call, so the head is already
    as stable as it can be."""

    HISTORY_LIMIT: ClassVar[int | None] = 40

    CHAT_HISTORY_STORE: ClassVar[type[ChatHistoryStore]] = InMemoryChatHistoryStore
    """Where this agent reads conversation history from.

    `base` owns no tables, so the default is the in-memory reference store. A
    service overrides this with its own persistent store — and must use the
    same one its routers use, or the two will resolve to different
    `ChatHistoryService` instances and see different history.
    """

    LLM_SERVICE: ClassVar[type[AbstractLLMService]] = OpenAIService
    """Which provider answers for this agent.

    Per agent rather than per application, so a cheap router and an expensive
    answerer can sit on different providers in one deployment.
    """

    MODEL_CONFIGURATIONS: ClassVar[ModelConfigurations | None] = None
    """Which model, when this agent wants one its provider's default is not.

    Left `None`, the agent resolves the provider's default instance — the same
    one `AuthenticatedLLM` installs a credentialled client over. Setting this
    opts out of that, because a configured service is keyed by its arguments.
    """

    def __init__(self) -> None:
        self._llm_service: AbstractLLMService = self._build_llm_service()
        self._prompt_service: PromptService = PromptService()
        self._chat_history_service: ChatHistoryService = ChatHistoryService(
            ChatHistoryConfig(), self.CHAT_HISTORY_STORE
        )
        self._logger: ConsoleLoggerService = ConsoleLoggerService()
        self._guardrail_pipeline: GuardrailPipeline | None = None

    @property
    def _guardrails(self) -> GuardrailPipeline:
        """Resolved on first use, not in `__init__`.

        An agent is a singleton and may be built before a service registers
        its settings; resolving here keeps construction free of that ordering.
        """
        if self._guardrail_pipeline is None:
            self._guardrail_pipeline = build_guardrail_pipeline()
        return self._guardrail_pipeline

    @classmethod
    def _build_llm_service(cls) -> AbstractLLMService:
        """Resolve this agent's provider.

        The no-argument call is deliberate when no configuration is set: a
        `KeyedSingleton` derives its key from its arguments, so passing an
        explicit `None` would resolve a different instance than the one
        `AuthenticatedLLM` installs, and the credential would be skipped.
        """
        if cls.MODEL_CONFIGURATIONS is None:
            return cls.LLM_SERVICE()
        return cls.LLM_SERVICE(cls.MODEL_CONFIGURATIONS)

    @property
    def _execution_context(self) -> ExecutionContext:
        """Resolve the current request's :class:`ExecutionContext` lazily.

        :class:`AbstractAgent` is a singleton (one instance per type),
        but :class:`ExecutionContext` is per-request scoped via
        ``Stored``. A property dispatches every access to the current
        scope's instance; storing it in ``__init__`` would freeze the
        first request's context across all subsequent requests.
        """
        return ExecutionContext()

    async def _pre_execute(self) -> None:
        """Optional setup hook. Default: no-op."""

    @abstractmethod
    async def _execute(self) -> None:
        """The agent's main work. Reads inputs from
        ``self._execution_context.flow_variables`` and writes outputs
        back. Lets :class:`LLMError` propagate — :meth:`arun` handles it.
        """

    async def _post_execute(self) -> None:
        """Optional teardown hook. Default: no-op. Runs only on success."""

    async def arun(self) -> None:
        await self._pre_execute()
        try:
            await self._execute()
        except LLMError as exc:
            canned_message = CannedMessages.from_llm_error(exc)
            self._logger.error(
                f"{type(self).__name__} failed: {type(exc).__name__}: {exc}"
            )
            if (
                self._execution_context.streaming
                and self._execution_context.stream_iterator is not None
            ):
                locale = self._execution_context.flow_variables.locale
                await self._execution_context.stream_iterator.asend(
                    StreamMessage(data=canned_message.value[locale])
                )
                await self._execution_context.stream_iterator.asend(StreamEnd())
            raise FlowAbortError(canned_message, cause=exc) from exc
        await self._post_execute()

    async def _aget_history(self) -> list[StoredMessage]:
        """Fetch the conversation's chat history, minus the trailing
        ``USER`` message.

        Honours :attr:`HISTORY_LIMIT` (most recent N stored messages,
        including the current user turn before it is sliced off).

        The router appends the current request's ``USER`` message to
        the conversation BEFORE the pipeline runs, so the latest
        ``USER`` in the store IS the current query. Each agent's
        prompt template already substitutes that query into the
        structured user message via ``{user_query}``; including the
        same text again as an inline history entry would duplicate it.
        We slice it off here.

        Concrete agents pass the result as ``history=`` to
        ``self._prompt_service.render(...)`` so the prompt's
        :class:`MessagesPlaceholder` can splice prior turns between
        the system instructions and the structured user content.
        """
        conversation_id = self._execution_context.flow_variables.conversation_id
        if conversation_id is None:
            return []
        all_messages = await self._chat_history_service.aget_messages(
            conversation_id,
            limit=self.HISTORY_LIMIT,
        )
        if all_messages and all_messages[-1].role == MessageRole.USER:
            return all_messages[:-1]
        return all_messages

    async def _acomplete(
        self,
        *,
        prompt: RenderedPrompt | None = None,
        messages: Sequence[StoredMessage] | None = None,
        tools: Sequence[Tool[BaseModel]] | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call the model with this agent's ClassVar defaults applied.

        Explicit kwargs always win, which is why the defaults go in through
        `setdefault`: naming them in the call as well would hand the service
        the same keyword twice over and raise before it ever ran.
        """
        self._apply_stream_iterator(kwargs)
        self._apply_defaults(kwargs, label)
        prompt, messages = await self._guarded_input(prompt, messages)
        stream = self._guarded_stream(kwargs)

        result = await self._llm_service.acomplete(
            prompt=prompt,
            messages=messages,
            tools=tools,
            **kwargs,
        )
        if stream is not None:
            if (blocked := await stream.afinish()) is not None:
                raise GuardrailBlocked(blocked)
            if stream.guard.assembled:
                return result.model_copy(update={"content": stream.guard.content})
        return await self._guarded_output(result)

    async def _aparse[T: BaseModel](
        self,
        schema: type[T],
        *,
        prompt: RenderedPrompt | None = None,
        messages: Sequence[StoredMessage] | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> ParsedResult[T]:
        """Same, for a call whose reply must fit `schema`.

        No stream iterator: a parsed call returns one object.
        """
        self._apply_defaults(kwargs, label)
        prompt, messages = await self._guarded_input(prompt, messages)

        return await self._llm_service.aparse(
            schema,
            prompt=prompt,
            messages=messages,
            **kwargs,
        )

    async def _guarded_input(
        self,
        prompt: RenderedPrompt | None,
        messages: Sequence[StoredMessage] | None,
    ) -> tuple[RenderedPrompt | None, Sequence[StoredMessage] | None]:
        """Check what a person wrote, before the model is shown it.

        Only user turns. Instructions and prior assistant replies are this
        service's own words — redacting them would quietly rewrite the prompt,
        and refusing them would refuse the service's own system message.
        """
        if not self._guardrails.applies_to(GuardrailStage.INPUT):
            return prompt, messages

        if prompt is not None:
            prompt = prompt.model_copy(
                update={"messages": await self._cleaned(prompt.messages)}
            )
        if messages is not None:
            messages = await self._cleaned(list(messages))
        return prompt, messages

    async def _cleaned[T: BaseModel](self, messages: Sequence[T]) -> list[T]:
        """The newest user turn may block; earlier ones are redacted or withheld."""
        newest = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if getattr(messages[i], "role", None) == MessageRole.USER
            ),
            None,
        )
        cleaned: list[T] = []
        for index, message in enumerate(messages):
            content = getattr(message, "content", None)
            if getattr(message, "role", None) != MessageRole.USER or not content:
                cleaned.append(message)
                continue
            report = await self._guardrails.arun_cached(content, GuardrailStage.INPUT)
            if report.blocked:
                if index == newest:
                    raise GuardrailBlocked(report)
                checked = BLOCKED_HISTORY_PLACEHOLDER
            else:
                checked = report.content
            cleaned.append(
                message
                if checked == content
                else message.model_copy(update={"content": checked})
            )
        return cleaned

    def _guarded_stream(self, kwargs: dict[str, Any]) -> GuardedStreamIterator | None:
        """Route a streamed reply through its guard, when anything would check it."""
        inner = kwargs.get("stream_iterator")
        if not isinstance(inner, StreamIterator):
            return None
        guard = build_guarded_stream(self._guardrails)
        if not guard.active:
            return None
        kwargs["stream_iterator"] = stream = GuardedStreamIterator(inner, guard)
        return stream

    async def _guarded_output(self, result: LLMResult) -> LLMResult:
        """Check a reply that was not streamed, before anyone sees it."""
        if not result.content:
            return result
        if not self._guardrails.applies_to(GuardrailStage.OUTPUT):
            return result
        checked = await self._guardrails.aenforce(result.content, GuardrailStage.OUTPUT)
        if checked == result.content:
            return result
        return result.model_copy(update={"content": checked})

    def _apply_defaults(self, kwargs: dict[str, Any], label: str | None) -> None:
        """This agent's ClassVars, for every knob the caller left alone."""
        kwargs.setdefault("max_output_tokens", self.MAX_OUTPUT_TOKENS)
        kwargs.setdefault("reasoning_effort", self.REASONING_EFFORT)
        kwargs.setdefault("cache_hint", self.CACHE_HINT)
        kwargs.setdefault("label", label or type(self).__name__)

    def _apply_stream_iterator(self, kwargs: dict[str, Any]) -> None:
        ctx = self._execution_context
        if "stream_iterator" not in kwargs:
            kwargs["stream_iterator"] = ctx.stream_iterator if ctx.streaming else None
