from typing import Any

from base.tools.errors import MissingExtraError

try:
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses import (
        ResponseInputItemParam,
        ResponseInputParam,
        ResponseUsage,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="Debug trace models") from exc
from pydantic import Field

from base.models.view_models import BaseModel

# Type aliases for the LLM-call payloads captured in the trace. The
# two API surfaces (Chat Completions and Responses) use different
# prompt and usage shapes; the union covers both. Re-exported here
# (rather than in ``services/llm/openai_service.py``) so :class:`DebugCall`
# can use them directly and the LLM service imports them too — single
# source of truth.

type LLMPromptItem = ChatCompletionMessageParam | ResponseInputItemParam
"""Element type of any captured prompt list. Used as the storage
element type on :class:`DebugCall.prompt` because Python's invariant
``list[T]`` makes a discriminated union of two concrete list types
(``list[A] | list[B]``) painful to round-trip through pydantic."""

type LLMPrompt = list[ChatCompletionMessageParam] | ResponseInputParam
"""Input type at the call boundary: callers pass one of the two
concrete list types. Pyright accepts a ``list[A]`` for the ``A`` arm
of the union without contortions; the storage layer relaxes to
``list[LLMPromptItem]`` because pydantic stores the merged element
type."""

type LLMResponseContent = str | dict[str, Any] | None
"""What we capture as the response payload: raw text from a
Completions / Responses call, the JSON-serialised parsed structured
output from a parse call, or ``None`` on error / no response."""

type LLMUsage = CompletionUsage | ResponseUsage | None
"""Token-usage payload returned by either API surface; ``None`` when
the call errored before producing a usage block."""


class DebugCall(BaseModel):
    """One LLM call captured at the OpenAIService boundary.

    ``prompt`` and ``response`` are the verbatim inputs / outputs the
    model saw. The ``label`` is usually the agent class name;
    per-item fanout calls suffix the item_id (e.g.
    ``XAgent:0001``) so the trace shows each fan-out
    call distinctly.
    """

    agent: str
    label: str
    model: str
    prompt: list[LLMPromptItem] = Field(default_factory=list)
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
