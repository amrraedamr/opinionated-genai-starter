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
    outcome = await ToolRunner(toolbox).arun(messages)
"""

from base.services.tool_calling.runner import ToolRunner, ToolRunResult
from base.services.tool_calling.toolbox import Tool, Toolbox, ToolError

__all__ = [
    "Tool",
    "ToolError",
    "ToolRunResult",
    "ToolRunner",
    "Toolbox",
]
