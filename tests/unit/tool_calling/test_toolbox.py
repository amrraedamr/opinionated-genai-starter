"""A `Toolbox` is one agent's allowance, not a global registry.

What it owes callers is a stable lookup and a safe dispatch: a name nobody
declared and a handler that blows up both come back as text the model can read,
because losing the turn over either would be worse than telling it what broke.
"""

from __future__ import annotations

import pytest

from base.models.view_models import BaseModel
from base.models.view_models.chat_history import StoredToolCall, StoredToolCallFunction
from base.services.tool_calling import Tool, Toolbox, ToolError


class Args(BaseModel):
    city: str


async def ok_handler(args: Args) -> str:
    return f"21 in {args.city}"


def tool(name: str) -> Tool[Args]:
    return Tool(name=name, description=name, parameters=Args, handler=ok_handler)


def call(name: str, arguments: str = '{"city": "Doha"}') -> StoredToolCall:
    return StoredToolCall(
        id=f"call_{name}",
        function=StoredToolCallFunction(name=name, arguments=arguments),
    )


def test_an_empty_toolbox_holds_nothing() -> None:
    toolbox = Toolbox()
    assert len(toolbox) == 0
    assert toolbox.names == []
    assert toolbox.tools == []
    assert "anything" not in toolbox


def test_names_are_sorted_so_a_prompt_is_stable_between_runs() -> None:
    toolbox = Toolbox(tool("zebra"), tool("alpha"), tool("middle"))
    assert toolbox.names == ["alpha", "middle", "zebra"]


def test_declaration_order_is_preserved_for_the_tools_themselves() -> None:
    toolbox = Toolbox(tool("zebra"), tool("alpha"))
    assert [t.name for t in toolbox.tools] == ["zebra", "alpha"]


def test_a_declared_tool_is_found_by_name() -> None:
    toolbox = Toolbox(tool("get_weather"))
    assert "get_weather" in toolbox
    assert len(toolbox) == 1


def test_duplicate_names_are_refused_at_construction() -> None:
    with pytest.raises(ToolError, match="Duplicate tool names"):
        Toolbox(tool("same"), tool("same"))


def test_the_duplicate_error_names_every_colliding_tool() -> None:
    with pytest.raises(ToolError) as caught:
        Toolbox(tool("a"), tool("a"), tool("b"), tool("b"), tool("c"))
    assert "'a'" in str(caught.value)
    assert "'b'" in str(caught.value)


async def test_invoking_a_declared_tool_returns_its_result() -> None:
    toolbox = Toolbox(tool("get_weather"))
    assert await toolbox.ainvoke(call("get_weather")) == "21 in Doha"


async def test_an_unknown_tool_is_reported_back_with_what_is_available() -> None:
    toolbox = Toolbox(tool("get_weather"))

    answer = await toolbox.ainvoke(call("get_stocks"))

    assert answer.startswith("Error: no tool named 'get_stocks'")
    assert "get_weather" in answer


async def test_an_empty_toolbox_still_answers_rather_than_raising() -> None:
    assert (await Toolbox().ainvoke(call("anything"))).startswith("Error: no tool")


async def test_bad_arguments_come_back_as_text_the_model_can_act_on() -> None:
    toolbox = Toolbox(tool("get_weather"))

    answer = await toolbox.ainvoke(call("get_weather", "{}"))

    assert answer.startswith("Error: get_weather rejected its arguments")


async def test_a_handler_that_raises_does_not_end_the_turn() -> None:
    async def explodes(args: Args) -> str:
        raise RuntimeError("upstream down")

    toolbox = Toolbox(
        Tool(name="boom", description="d", parameters=Args, handler=explodes)
    )

    answer = await toolbox.ainvoke(call("boom"))

    assert answer == "Error: boom failed: RuntimeError: upstream down"
