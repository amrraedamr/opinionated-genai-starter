from typing import ClassVar

from base.services.agents import AbstractAgent
from base.services.store.chat_history import ChatHistoryStore
from base.services.tool_calling.runner import ToolRunner
from chat_api.mcp import MCP
from chat_api.stores.chat_history.sql import SqlChatHistoryStore


class TestAgent(AbstractAgent):
    """Test agent for development and debugging, with every configured MCP server's tools."""

    PROMPT_KEY: ClassVar[str] = "test_prompt"
    MAX_OUTPUT_TOKENS: ClassVar[int] = 500
    MAX_TOOL_ROUNDS: ClassVar[int] = 5

    # Must match what the routers use, or the agent reads a different history.
    CHAT_HISTORY_STORE: ClassVar[type[ChatHistoryStore]] = SqlChatHistoryStore

    async def _execute(self) -> None:
        context = self._execution_context
        rendered_prompt = self._prompt_service.render(
            self.PROMPT_KEY,
            history=await self._aget_history(),
            user_query=context.flow_variables.user_query,
        )
        turn = await ToolRunner(
            await MCP.atoolbox(),
            max_rounds=self.MAX_TOOL_ROUNDS,
            guardrails=self._guardrails,
            complete=self._acomplete,
        ).arun(
            prompt=rendered_prompt,
            conversation_id=context.flow_variables.conversation_id,
            stream_iterator=context.stream_iterator if context.streaming else None,
        )
        context.flow_variables.final_answer = (turn.result.content or "").strip()
