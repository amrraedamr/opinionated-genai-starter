"""The tools this service offers. Replace these with your own; the wiring stays as it is.

One tool per way the library describes a tool: by a Pydantic model, which most
are, and by raw JSON Schema, which is how a tool arrives from elsewhere. `whoami`
reads who called it, the same `TenantContext` a route would see, and `echo` returns
text the caller chose, which is what the caller's `TOOL_RESULT` guardrail checks.
"""

from datetime import UTC, datetime

from pydantic import Field

from base.models.view_models import BaseModel, JSONObject
from base.services.tool_calling import Tool, Toolbox
from base.tools.tenancy import CallerContext, TenantContext


class AddArguments(BaseModel):
    """Two numbers to add."""

    a: float
    b: float


async def add(arguments: AddArguments) -> str:
    return str(arguments.a + arguments.b)


class WordCountArguments(BaseModel):
    """Text to count the words of."""

    text: str = Field(max_length=10_000)


async def word_count(arguments: WordCountArguments) -> str:
    return str(len(arguments.text.split()))


class EchoArguments(BaseModel):
    """Text to hand straight back."""

    text: str = Field(max_length=10_000)


async def echo(arguments: EchoArguments) -> str:
    return arguments.text


NO_ARGUMENTS: JSONObject = {"type": "object", "properties": {}}


async def utc_now(arguments: JSONObject) -> str:
    del arguments
    return datetime.now(UTC).isoformat()


async def whoami(arguments: JSONObject) -> str:
    del arguments
    tenant = TenantContext.get() or "none"
    return f"caller={CallerContext.get()} tenant={tenant}"


toolbox = Toolbox(
    Tool(
        name="add", description="Add two numbers.", parameters=AddArguments, handler=add
    ),
    Tool(
        name="word_count",
        description="Count the words in a piece of text.",
        parameters=WordCountArguments,
        handler=word_count,
    ),
    Tool(
        name="echo",
        description="Return the given text unchanged.",
        parameters=EchoArguments,
        handler=echo,
    ),
    Tool.from_json_schema(
        name="utc_now",
        description="The current time, in UTC, as ISO 8601.",
        schema=NO_ARGUMENTS,
        handler=utc_now,
    ),
    Tool.from_json_schema(
        name="whoami",
        description="Who this call is running as, and for which tenant.",
        schema=NO_ARGUMENTS,
        handler=whoami,
    ),
)
