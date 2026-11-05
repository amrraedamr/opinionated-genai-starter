"""What a request did, for looking at afterwards.

Accounting lives in `UsageRecord`: tokens, cost, duration, how many attempts a
call took and what each failed one said. None of that is repeated here. What
this adds is the part a usage ledger should not carry — the verbatim prompt and
response — because those are large, often sensitive, and only wanted when
someone is actually debugging. `call_id` ties the two together.
"""

from typing import Any
from uuid import UUID

from pydantic import Field

from base.models.view_models import BaseModel

type LLMResponseContent = str | dict[str, Any] | None
"""What we capture as the response payload: raw text, the JSON-serialised
parsed output of a structured call, or `None` on error."""


class DebugCall(BaseModel):
    """The payloads of one model call, captured at the adapter boundary.

    `label` is usually the agent class name; per-item fanout calls suffix the
    item id (`XAgent:0001`) so the trace shows each one distinctly.
    """

    call_id: UUID
    """Matches the `UsageRecord` for the same call, which holds its numbers."""

    label: str
    model: str
    prompt: list[dict[str, Any]] = Field(default_factory=list)
    response: LLMResponseContent = None


class FlowEvent(BaseModel):
    """One pipeline-level event: a stage starting or ending, a branch taken, a
    filter's result count. Lighter than a `DebugCall`, and written by services
    that make no model call at all."""

    elapsed_ms: int
    stage: str
    detail: dict[str, Any] = Field(default_factory=dict)


class DebugTrace(BaseModel):
    """Per-request bundle of captured calls and flow events.

    Populated only when a request asks for it, since capturing prompts costs
    memory and exposes content that normal telemetry should not hold.
    """

    started_at: float = 0.0
    calls: list[DebugCall] = Field(default_factory=list)
    events: list[FlowEvent] = Field(default_factory=list)
