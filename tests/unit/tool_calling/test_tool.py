"""A `Tool`'s Pydantic model is both the schema shown and the validator used.

That is the whole point of declaring one: a malformed call is rejected before
the handler sees it, so a handler never has to defend itself against the
model's typos.
"""

from __future__ import annotations

from typing import Any

import pytest

from base.models.view_models import BaseModel
from base.services.tool_calling import Tool, ToolError


class Weather(BaseModel):
    city: str
    units: str = "c"


async def weather_handler(args: Weather) -> str:
    return f"21{args.units} in {args.city}"


def build(name: str = "get_weather") -> Tool[Weather]:
    return Tool(
        name=name,
        description="Current weather for a city.",
        parameters=Weather,
        handler=weather_handler,
    )


def test_a_tool_carries_the_name_and_description_a_model_is_shown() -> None:
    tool = build()
    assert tool.name == "get_weather"
    assert tool.description == "Current weather for a city."


def test_the_json_schema_comes_from_the_parameters_model() -> None:
    schema = build().json_schema

    assert schema == Weather.model_json_schema()
    assert schema["properties"]["city"]["type"] == "string"
    assert schema["required"] == ["city"]
    assert schema["properties"]["units"]["default"] == "c"


def test_a_tool_is_frozen_so_a_toolbox_cannot_be_rewritten_under_an_agent() -> None:
    import dataclasses

    tool = build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        tool.name = "something_else"  # type: ignore[misc]


async def test_valid_arguments_reach_the_handler_parsed() -> None:
    seen: list[Weather] = []

    async def recording(args: Weather) -> str:
        seen.append(args)
        return "done"

    tool = Tool(name="t", description="d", parameters=Weather, handler=recording)

    assert await tool.ainvoke('{"city": "Doha", "units": "f"}') == "done"
    assert seen == [Weather(city="Doha", units="f")]


async def test_a_default_is_applied_when_the_model_omits_it() -> None:
    assert await build().ainvoke('{"city": "Doha"}') == "21c in Doha"


async def test_a_missing_required_argument_never_reaches_the_handler() -> None:
    ran: list[Any] = []

    async def never(args: Weather) -> str:
        ran.append(args)
        return "ran"

    tool = Tool(name="t", description="d", parameters=Weather, handler=never)

    with pytest.raises(ToolError, match="rejected its arguments"):
        await tool.ainvoke("{}")
    assert ran == []


async def test_a_wrongly_typed_argument_is_rejected() -> None:
    with pytest.raises(ToolError, match="rejected its arguments"):
        await build().ainvoke('{"city": 42}')


async def test_arguments_that_are_not_json_are_rejected() -> None:
    with pytest.raises(ToolError, match="rejected its arguments"):
        await build().ainvoke("city=Doha")


async def test_the_error_names_the_tool_that_refused() -> None:
    with pytest.raises(ToolError) as caught:
        await build("lookup_city").ainvoke("{}")
    assert "lookup_city" in str(caught.value)


async def test_a_non_string_result_is_serialized_for_the_model() -> None:
    class Empty(BaseModel):
        pass

    async def structured(args: Empty) -> dict[str, int]:
        return {"celsius": 21}

    tool = Tool(name="t", description="d", parameters=Empty, handler=structured)

    assert await tool.ainvoke("{}") == '{"celsius": 21}'


async def test_a_string_result_is_passed_through_unquoted() -> None:
    answer = await build().ainvoke('{"city": "Doha"}')

    assert answer == "21c in Doha"
    assert not answer.startswith('"')
