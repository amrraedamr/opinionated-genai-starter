"""Tool declarations and the dispatch behind them."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredToolCall
from base.tools.serializers import serialize


class ToolError(Exception):
    """A tool could not be run as asked."""


@dataclass(frozen=True, slots=True)
class Tool[T: BaseModel]:
    """One callable an LLM may invoke, described by its argument model.

    `parameters` is both the schema the model is shown and the validator the
    model's arguments are checked against, so the two cannot drift apart.
    """

    name: str
    description: str
    parameters: type[T]
    handler: Callable[[T], Awaitable[Any]]

    @property
    def schema(self) -> dict[str, Any]:
        """The tool as the chat completions API wants to see it."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_json_schema(),
            },
        }

    @property
    def responses_schema(self) -> dict[str, Any]:
        """The same tool, flattened the way the Responses API wants it."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.model_json_schema(),
        }

    async def ainvoke(self, arguments: str) -> str:
        """Validate JSON `arguments`, run the handler, and render the result.

        Raises `ToolError` when the arguments do not fit `parameters`. The
        caller decides whether that goes back to the model or aborts the turn.
        """
        try:
            parsed = self.parameters.model_validate_json(arguments)
        except ValidationError as exc:
            raise ToolError(f"{self.name} rejected its arguments: {exc}") from exc

        result = await self.handler(parsed)
        return result if isinstance(result, str) else serialize(result)


class Toolbox:
    """The tools one agent may call.

    Deliberately not a global registry: which tools an agent is allowed to
    reach is part of that agent's definition, not ambient state.
    """

    def __init__(self, *tools: Tool[Any]) -> None:
        duplicates = {t.name for t in tools if sum(o.name == t.name for o in tools) > 1}
        if duplicates:
            raise ToolError(f"Duplicate tool names: {sorted(duplicates)}")
        self._tools: dict[str, Tool[Any]] = {tool.name: tool for tool in tools}

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Every tool, in the shape the chat completions API expects."""
        return [tool.schema for tool in self._tools.values()]

    @property
    def responses_schemas(self) -> list[dict[str, Any]]:
        """Every tool, in the shape the Responses API expects."""
        return [tool.responses_schema for tool in self._tools.values()]

    async def ainvoke(self, call: StoredToolCall) -> str:
        """Run the tool a model asked for and return what it should see.

        A handler that raises is reported back as text rather than propagated:
        a model that called a tool wrongly can usually recover if it is told
        how, and an exception here would lose the rest of the turn.
        """
        tool = self._tools.get(call.function.name)
        if tool is None:
            return f"Error: no tool named {call.function.name!r}. Available: {self.names}."

        try:
            return await tool.ainvoke(call.function.arguments)
        except ToolError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: {call.function.name} failed: {type(exc).__name__}: {exc}"
