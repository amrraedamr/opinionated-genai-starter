from typing import ClassVar

from base.services.agents import AbstractAgent
from base.services.llm import OpenAIMessageAdapter
from base.services.store.chat_history import ChatHistoryStore
from chat_api.stores.chat_history.sql import SqlChatHistoryStore


class TestAgent(AbstractAgent):
    """Test agent for development and debugging."""

    PROMPT_KEY: ClassVar[str] = "test_prompt"
    MAX_OUTPUT_TOKENS: ClassVar[int] = 500

    # Must match what the routers use, or the agent reads a different history.
    CHAT_HISTORY_STORE: ClassVar[type[ChatHistoryStore]] = SqlChatHistoryStore

    async def _execute(self) -> None:
        rendered_prompt = self._prompt_service.render(
            self.PROMPT_KEY,
            history=await self._aget_history(),
            user_query=self._execution_context.flow_variables.user_query,
        )
        result = await self._aresponses(
            OpenAIMessageAdapter.to_responses_input(prompt=rendered_prompt),
        )
        self._execution_context.flow_variables.final_answer = (
            result.content or ""
        ).strip()
