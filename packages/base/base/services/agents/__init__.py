from abc import abstractmethod
from typing import Any, ClassVar

from base.tools.errors import MissingExtraError

try:
    from openai.types.responses import ResponseInputParam
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="AbstractAgent") from exc

from base.models.definitions.canned_messages import CannedMessages
from base.models.definitions.message import MessageRole
from base.models.definitions.model import ReasoningEffort
from base.models.view_models.chat_history import ChatHistoryConfig, StoredMessage
from base.models.view_models.flow import FlowAbortError
from base.models.view_models.llm_exception import LLMError
from base.models.view_models.llm_result import ResponseResult
from base.models.view_models.streams import StreamEnd, StreamMessage
from base.services.chat_history import ChatHistoryService
from base.services.common.meta import SingletonABC
from base.services.execution_context import ExecutionContext
from base.services.llm import OpenAIService
from base.services.logging.console import ConsoleLoggerService
from base.services.prompt.service import PromptService
from base.services.store.chat_history import ChatHistoryStore
from base.services.store.chat_history.memory import InMemoryChatHistoryStore


class AbstractAgent(metaclass=SingletonABC):
    """Template-method base for every LLM agent in a chat / pipeline turn.

    All three hooks are async and take no arguments — agents read from
    and write to the shared :class:`ExecutionContext.flow_variables`.

    On a fatal LLM error, ``arun`` maps the exception to a
    :class:`CannedMessages`, streams it inline (when the request is
    streaming), and raises :class:`FlowAbortError` for the pipeline to
    catch.

    Per-call debug capture lives inside :class:`OpenAIService` itself
    (try/finally around every ``aresponses`` / ``aparse_responses``
    call). Concrete agents pass ``label=...`` to the LLM service so the
    captured :class:`DebugCall` carries a meaningful identifier — by
    convention ``label=type(self).__name__`` for single-shot agents and
    ``label=f"{type(self).__name__}:{item_id}"`` when fanout calls need
    a stable per-item discriminator.
    """

    PROMPT_KEY: ClassVar[str] = ""
    MAX_OUTPUT_TOKENS: ClassVar[int] = 1000
    REASONING_EFFORT: ClassVar[ReasoningEffort | None] = None
    HISTORY_LIMIT: ClassVar[int | None] = 40

    CHAT_HISTORY_STORE: ClassVar[type[ChatHistoryStore]] = InMemoryChatHistoryStore
    """Where this agent reads conversation history from.

    `base` owns no tables, so the default is the in-memory reference store. A
    service overrides this with its own persistent store — and must use the
    same one its routers use, or the two will resolve to different
    `ChatHistoryService` instances and see different history.
    """

    def __init__(self) -> None:
        self._llm_service: OpenAIService = OpenAIService()
        self._prompt_service: PromptService = PromptService()
        self._chat_history_service: ChatHistoryService = ChatHistoryService(
            ChatHistoryConfig(), self.CHAT_HISTORY_STORE
        )
        self._logger: ConsoleLoggerService = ConsoleLoggerService()

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

    async def _aresponses(
        self,
        input_: ResponseInputParam,
        *,
        label: str | None = None,
        **kwargs: Any,
    ) -> ResponseResult:
        """Call :meth:`OpenAIService.aresponses` with agent ClassVar defaults.

        Applies ``MAX_OUTPUT_TOKENS``, ``REASONING_EFFORT``, and the
        request's stream iterator unless the caller overrides them.
        Explicit kwargs always win.
        """
        kwargs.setdefault("max_output_tokens", self.MAX_OUTPUT_TOKENS)
        if self.REASONING_EFFORT is not None:
            kwargs.setdefault("reasoning", {"effort": self.REASONING_EFFORT})

        ctx = self._execution_context
        if "stream_iterator" not in kwargs:
            kwargs["stream_iterator"] = ctx.stream_iterator if ctx.streaming else None

        return await self._llm_service.aresponses(
            input_,
            label=label or type(self).__name__,
            **kwargs,
        )
