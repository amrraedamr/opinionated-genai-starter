"""Tool declarations and the dispatch behind them."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import ValidationError

from base.models.view_models import BaseModel, JSONObject
from base.models.view_models.chat_history import StoredToolCall
from base.tools.errors import MissingExtraError
from base.tools.serializers import serialize

if TYPE_CHECKING:
    from jsonschema.protocols import Validator


T_co = TypeVar("T_co", bound=BaseModel, covariant=True)
"""Covariant so a `Tool[Weather]` is a `Tool[BaseModel]`; sound because a tool is frozen."""


class ToolError(Exception):
    """A tool could not be run as asked."""


@dataclass(frozen=True, slots=True)
class Tool(Generic[T_co]):
    """One callable an LLM may invoke, described by its argument model.

    `parameters` is both the schema the model is shown and the validator the
    model's arguments are checked against, so the two cannot drift apart. A
    tool described elsewhere, such as by an MCP server, has raw JSON Schema and
    no model; `from_json_schema` builds one, so adapters see a single type.
    """

    name: str
    description: str
    parameters: type[T_co] | None
    handler: Callable[..., Awaitable[object]]
    """Called with the parsed model, or the raw object when described by schema."""
    schema: JSONObject | None = field(default=None)

    def __post_init__(self) -> None:
        if (self.parameters is None) == (self.schema is None):
            raise ToolError(
                f"{self.name} needs exactly one of a parameters model or a JSON Schema"
            )

    @classmethod
    def from_json_schema(
        cls,
        *,
        name: str,
        description: str,
        schema: JSONObject,
        handler: Callable[[JSONObject], Awaitable[object]],
    ) -> "Tool[BaseModel]":
        """A tool whose arguments are described by raw JSON Schema rather than a model."""
        cls._schema_validator(schema)
        return Tool[BaseModel](
            name=name,
            description=description,
            parameters=None,
            handler=handler,
            schema=schema,
        )

    @property
    def json_schema(self) -> JSONObject:
        """The argument schema as JSON Schema. Adapters reshape it per provider."""
        if self.schema is not None:
            return self.schema
        assert self.parameters is not None
        return self.parameters.model_json_schema()

    async def ainvoke(self, arguments: str) -> str:
        """Validate JSON `arguments`, run the handler, and render the result.

        Raises `ToolError` when the arguments do not fit the declared schema. The
        caller decides whether that goes back to the model or aborts the turn.
        """
        parsed = self._parse(arguments)
        result = await self.handler(parsed)
        return result if isinstance(result, str) else serialize(result)

    def _parse(self, arguments: str) -> T_co | JSONObject:
        if self.parameters is not None:
            try:
                return self.parameters.model_validate_json(arguments)
            except ValidationError as exc:
                raise ToolError(f"{self.name} rejected its arguments: {exc}") from exc
        return self._parse_against_schema(arguments)

    def _parse_against_schema(self, arguments: str) -> JSONObject:
        try:
            decoded = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolError(f"{self.name} rejected its arguments: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ToolError(f"{self.name} rejected its arguments: not an object")
        assert self.schema is not None
        errors = sorted(
            self._schema_validator(self.schema).iter_errors(decoded),
            key=lambda error: list(error.path),
        )
        if errors:
            detail = "; ".join(error.message for error in errors)
            raise ToolError(f"{self.name} rejected its arguments: {detail}")
        return decoded

    @staticmethod
    def _schema_validator(schema: JSONObject) -> "Validator":
        """A validator for `schema`, refusing a schema that is itself malformed."""
        try:
            from jsonschema import SchemaError, validators
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise MissingExtraError(
                "tools", "jsonschema", feature="Tool.from_json_schema"
            ) from exc
        validator_class = validators.validator_for(schema)
        try:
            validator_class.check_schema(schema)
        except SchemaError as exc:
            raise ToolError(f"invalid JSON Schema: {exc.message}") from exc
        return validator_class(schema)


class Toolbox:
    """The tools one agent may call.

    Deliberately not a global registry: which tools an agent is allowed to
    reach is part of that agent's definition, not ambient state.
    """

    def __init__(self, *tools: Tool[BaseModel]) -> None:
        """A registry of tools, each with its own argument model."""
        duplicates = {t.name for t in tools if sum(o.name == t.name for o in tools) > 1}
        if duplicates:
            raise ToolError(f"Duplicate tool names: {sorted(duplicates)}")
        self._tools: dict[str, Tool[BaseModel]] = {tool.name: tool for tool in tools}

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    @property
    def tools(self) -> list[Tool[BaseModel]]:
        """Every tool, as declared. Shaping them for a provider is adapter work."""
        return list(self._tools.values())

    async def ainvoke(self, call: StoredToolCall) -> str:
        """Run the tool a model asked for and return what it should see.

        A handler that raises is reported back as text rather than propagated:
        a model that called a tool wrongly can usually recover if it is told
        how, and an exception here would lose the rest of the turn.
        """
        tool = self._tools.get(call.function.name)
        if tool is None:
            return (
                f"Error: no tool named {call.function.name!r}. Available: {self.names}."
            )

        try:
            return await tool.ainvoke(call.function.arguments)
        except ToolError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: {call.function.name} failed: {type(exc).__name__}: {exc}"
