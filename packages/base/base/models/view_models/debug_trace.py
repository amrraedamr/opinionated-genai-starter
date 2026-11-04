from typing import Any

from pydantic import Field

from base.models.view_models import BaseModel

type LLMResponseContent = str | dict[str, Any] | None
"""What we capture as the response payload: raw text, the JSON-serialised
parsed output of a structured call, or `None` on error."""


class DebugCall(BaseModel):
    """One LLM call captured at the adapter boundary.

    ``prompt`` and ``response`` are the verbatim inputs / outputs the
    model saw. The ``label`` is usually the agent class name;
    per-item fanout calls suffix the item_id (e.g.
    ``XAgent:0001``) so the trace shows each fan-out
    call distinctly.
    """

    agent: str
    label: str
    model: str
    prompt: list[dict[str, Any]] = Field(default_factory=list)
    response: LLMResponseContent = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    error: str | None = None


class FlowEvent(BaseModel):
    """One pipeline-level event (stage start, stage end, identify path
    chosen, filter result counts, etc.). Lighter than ``DebugCall``;
    used by the pipeline + non-agent services to leave breadcrumbs."""

    elapsed_ms: int
    stage: str
    detail: dict[str, Any] = Field(default_factory=dict)


class DebugTrace(BaseModel):
    """Per-request bundle of LLM calls + flow events. Always populated
    on :class:`ExecutionContext`. Persisted in chat-history metadata
    under ``metadata.debug`` and optionally returned inline on the
    response when ``?debug=true``."""

    started_at: float = 0.0
    calls: list[DebugCall] = Field(default_factory=list)
    events: list[FlowEvent] = Field(default_factory=list)
