"""Declaring tools an LLM may call, and running the turn that calls them.

A tool is a Python callable plus the Pydantic model describing its arguments.
That model is the single source of truth: it generates the JSON schema the
model sees and validates the arguments the model sends back, so a malformed
call is caught here rather than inside the handler.

    class Weather(BaseModel):
        city: str

    async def get_weather(args: Weather) -> str:
        return f"It is cold in {args.city}."

    toolbox = Toolbox(
        Tool(
            name="get_weather",
            description="Current weather for a city.",
            parameters=Weather,
            handler=get_weather,
        )
    )
    from base.services.tool_calling.runner import ToolRunner

    outcome = await ToolRunner(toolbox).arun(messages)

`ToolRunner` is imported from its own module rather than re-exported here.
Declaring a tool sits below the LLM adapters, which render a `Tool` into
whatever shape their provider wants; running a turn sits above them, because
it needs a model to call. Re-exporting the runner would drag the upper layer
into every import of the lower one, and the two would form a cycle.
"""

from base.services.tool_calling.toolbox import Tool, Toolbox, ToolError

__all__ = [
    "Tool",
    "ToolError",
    "Toolbox",
]
