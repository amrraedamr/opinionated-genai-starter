"""Tools described by raw JSON Schema, as an MCP server hands them over.

An MCP tool has no Pydantic model behind it: the server sends a JSON Schema
and nothing else. Such a tool has to render on every provider, validate what a
model sends against that schema, and refuse a schema that is itself broken —
the same three promises a model-backed tool makes.
"""

import pytest

from base.models.view_models import BaseModel, JSONObject
from base.models.view_models.chat_history import StoredToolCall, StoredToolCallFunction
from base.services.llm.anthropic.adapter import AnthropicMessageAdapter
from base.services.llm.gemini.adapter import GeminiContentAdapter
from base.services.llm.openai.adapter import OpenAIMessageAdapter
from base.services.tool_calling import Tool, Toolbox, ToolError

SCHEMA: JSONObject = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer", "minimum": 1, "maximum": 7},
    },
    "required": ["city"],
}


async def forecast(arguments: JSONObject) -> str:
    return f"{arguments['city']} for {arguments.get('days', 1)} day(s)"


def schema_tool(schema: JSONObject = SCHEMA, name: str = "forecast") -> Tool[BaseModel]:
    return Tool.from_json_schema(
        name=name,
        description="Forecast for a city.",
        schema=schema,
        handler=forecast,
    )


class City(BaseModel):
    city: str


async def model_handler(arguments: City) -> str:
    return arguments.city


# Construction


def test_a_schema_tool_shows_the_model_exactly_the_schema_it_was_given() -> None:
    assert schema_tool().json_schema == SCHEMA


def test_a_schema_tool_has_no_parameters_model() -> None:
    assert schema_tool().parameters is None


def test_a_malformed_schema_is_refused_when_the_tool_is_built() -> None:
    with pytest.raises(ToolError, match="invalid JSON Schema"):
        schema_tool({"type": "object", "properties": {"city": {"type": 7}}})


def test_a_tool_needs_one_description_of_its_arguments_not_two() -> None:
    with pytest.raises(ToolError, match="exactly one"):
        Tool(
            name="both",
            description="Both kinds.",
            parameters=City,
            handler=model_handler,
            schema=SCHEMA,
        )


def test_a_tool_needs_one_description_of_its_arguments_not_none() -> None:
    with pytest.raises(ToolError, match="exactly one"):
        Tool(name="neither", description="Neither.", parameters=None, handler=forecast)


def test_a_model_backed_tool_is_unchanged() -> None:
    tool = Tool(
        name="city", description="A city.", parameters=City, handler=model_handler
    )
    assert tool.json_schema == City.model_json_schema()


# Invocation


async def test_arguments_that_fit_the_schema_reach_the_handler_as_a_dict() -> None:
    assert await schema_tool().ainvoke('{"city": "Cairo", "days": 3}') == (
        "Cairo for 3 day(s)"
    )


async def test_a_missing_required_argument_is_refused() -> None:
    with pytest.raises(ToolError, match="'city' is a required property"):
        await schema_tool().ainvoke('{"days": 3}')


async def test_an_argument_outside_its_bounds_is_refused() -> None:
    with pytest.raises(ToolError, match="maximum"):
        await schema_tool().ainvoke('{"city": "Cairo", "days": 30}')


async def test_an_argument_of_the_wrong_type_is_refused() -> None:
    with pytest.raises(ToolError, match="is not of type 'integer'"):
        await schema_tool().ainvoke('{"city": "Cairo", "days": "three"}')


async def test_every_violation_is_reported_not_only_the_first() -> None:
    with pytest.raises(ToolError) as caught:
        await schema_tool().ainvoke('{"days": "three"}')
    assert "required" in str(caught.value)
    assert "integer" in str(caught.value)


async def test_arguments_that_are_not_json_are_refused() -> None:
    with pytest.raises(ToolError, match="rejected its arguments"):
        await schema_tool().ainvoke("{not json")


async def test_arguments_that_are_not_an_object_are_refused() -> None:
    with pytest.raises(ToolError, match="not an object"):
        await schema_tool().ainvoke('["Cairo"]')


async def test_empty_arguments_are_read_as_an_empty_object() -> None:
    async def echo(arguments: JSONObject) -> str:
        return repr(arguments)

    lenient = Tool.from_json_schema(
        name="echo",
        description="Echo.",
        schema={"type": "object", "properties": {}},
        handler=echo,
    )
    assert await lenient.ainvoke("") == "{}"


async def test_a_schema_tool_runs_through_a_toolbox_like_any_other() -> None:
    toolbox = Toolbox(schema_tool())
    call = StoredToolCall(
        id="call_1",
        function=StoredToolCallFunction(name="forecast", arguments='{"city": "Doha"}'),
    )
    assert await toolbox.ainvoke(call) == "Doha for 1 day(s)"


async def test_a_toolbox_reports_a_schema_violation_back_as_text() -> None:
    toolbox = Toolbox(schema_tool())
    call = StoredToolCall(
        id="call_1", function=StoredToolCallFunction(name="forecast", arguments="{}")
    )
    assert (await toolbox.ainvoke(call)).startswith("Error:")


# Rendering on every provider


def test_openai_chat_completions_renders_the_raw_schema() -> None:
    rendered = OpenAIMessageAdapter.to_chat_completions_tools([schema_tool()])
    assert rendered[0]["function"]["parameters"] == SCHEMA


def test_openai_responses_renders_the_raw_schema() -> None:
    rendered = OpenAIMessageAdapter.to_responses_tools([schema_tool()])
    assert rendered[0]["parameters"] == SCHEMA


def test_anthropic_renders_the_raw_schema() -> None:
    rendered = AnthropicMessageAdapter.to_tools([schema_tool()])
    assert rendered[0]["input_schema"] == SCHEMA


def test_gemini_renders_a_raw_schema_rather_than_reaching_for_a_model() -> None:
    """Gemini read `tool.parameters` directly, which a schema tool does not have."""
    rendered = GeminiContentAdapter.to_tools([schema_tool()])
    declaration = rendered[0]["function_declarations"][0]
    assert declaration["parameters"]["properties"]["city"] == {"type": "string"}
    assert declaration["parameters"]["required"] == ["city"]


def test_gemini_still_strips_what_it_does_not_accept_from_a_raw_schema() -> None:
    noisy = {**SCHEMA, "title": "Forecast", "additionalProperties": False}
    rendered = GeminiContentAdapter.to_tools([schema_tool(noisy)])
    declaration = rendered[0]["function_declarations"][0]["parameters"]
    assert "title" not in declaration
    assert "additionalProperties" not in declaration


def test_both_kinds_of_tool_render_side_by_side() -> None:
    model_tool = Tool(
        name="city", description="A city.", parameters=City, handler=model_handler
    )
    rendered = GeminiContentAdapter.to_tools([model_tool, schema_tool()])
    names = [d["name"] for d in rendered[0]["function_declarations"]]
    assert names == ["city", "forecast"]
