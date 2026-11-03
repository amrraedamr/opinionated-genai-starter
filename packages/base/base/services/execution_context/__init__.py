import time
from typing import Any

from base.models.view_models.debug_trace import DebugTrace, FlowEvent
from base.models.view_models.execution_context import FlowVariables
from base.services.common.meta import Stored
from base.tools.stream import StreamIterator


class ExecutionContext(metaclass=Stored):
    """Per-request data container shared by every agent and service in
    a chat / pipeline turn.

    One instance per ``scope_id`` (set by the ``register_request``
    middleware in :mod:`base.main`). The middleware also tears the
    scope down on response — except for ``text/event-stream`` responses
    where the SSE drains after the handler returns; in that case the
    scope persists for the duration of the stream and is cleaned up
    when the process ends. (Pre-existing behaviour; not addressed
    here.)

    Streaming flags and the debug trace are kept on the context (not
    on :class:`FlowVariables`) because they're orthogonal to the
    typed flow data — they're transport / observability concerns,
    not flow state.
    """

    def __init__(self) -> None:
        self.flow_variables: FlowVariables = FlowVariables()
        self.streaming: bool = False
        self.stream_iterator: StreamIterator | None = None
        self.debug_trace: DebugTrace = DebugTrace()

    def set_flow_variable(self, key: str, value: Any) -> None:
        setattr(self.flow_variables, key, value)

    def get_flow_variable(self, key: str, default: Any = None) -> Any:
        return getattr(self.flow_variables, key, default)

    def get_flow_variables(self) -> FlowVariables:
        return self.flow_variables

    def start_debug_trace(self) -> None:
        """Stamp the trace's start time. Call once at the start of a
        pipeline / agent turn so subsequent ``record_event`` calls can
        compute monotonic ``elapsed_ms``."""
        self.debug_trace.started_at = time.perf_counter()

    def record_event(self, stage: str, **detail: Any) -> FlowEvent:
        """Append a :class:`FlowEvent` to the trace with the elapsed
        ms since :meth:`start_debug_trace`. Returns the event so the
        caller can log it however it wants."""
        elapsed_ms = (
            int((time.perf_counter() - self.debug_trace.started_at) * 1000)
            if self.debug_trace.started_at
            else 0
        )
        event = FlowEvent(elapsed_ms=elapsed_ms, stage=stage, detail=detail)
        self.debug_trace.events.append(event)
        return event
