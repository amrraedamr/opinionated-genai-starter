"""`ExecutionContext` is one request's shared state, and only one request's.

It is a `Stored` singleton, so what this pins is scope: two scopes get two
containers, and the debug trace stamps elapsed time against the start the
scope recorded rather than against process start.
"""

from __future__ import annotations

import uuid

from base.models.definitions.database import Locale
from base.models.view_models.chat_history import ConversationId
from base.models.view_models.execution_context import FlowVariables
from base.services.common.meta import Stored, scope_id
from base.services.execution_context import ExecutionContext


def test_one_scope_sees_one_context() -> None:
    assert ExecutionContext() is ExecutionContext()


def test_another_scope_sees_another_context() -> None:
    first = ExecutionContext()
    token = scope_id.set(uuid.uuid4())
    try:
        assert ExecutionContext() is not first
    finally:
        scope_id.reset(token)


def test_deleting_a_scope_discards_its_context() -> None:
    token = scope_id.set(uuid.uuid4())
    try:
        original = ExecutionContext()
        Stored.delete_scope()
        assert ExecutionContext() is not original
    finally:
        scope_id.reset(token)


def test_a_fresh_context_is_not_streaming_and_carries_empty_flow_variables() -> None:
    token = scope_id.set(uuid.uuid4())
    try:
        context = ExecutionContext()
        assert context.streaming is False
        assert context.stream_iterator is None
        assert context.flow_variables == FlowVariables()
        assert context.flow_variables.locale is Locale.EN
    finally:
        scope_id.reset(token)


def test_flow_variables_are_set_and_read_by_name() -> None:
    token = scope_id.set(uuid.uuid4())
    try:
        context = ExecutionContext()
        conversation = ConversationId(uuid.uuid4())
        context.set_flow_variable("conversation_id", conversation)

        assert context.get_flow_variable("conversation_id") == conversation
        assert context.get_flow_variables() is context.flow_variables
    finally:
        scope_id.reset(token)


def test_an_unknown_flow_variable_falls_back_to_the_default() -> None:
    token = scope_id.set(uuid.uuid4())
    try:
        assert ExecutionContext().get_flow_variable("nothing", "fallback") == "fallback"
    finally:
        scope_id.reset(token)


def test_an_event_recorded_before_the_trace_starts_has_no_elapsed_time() -> None:
    token = scope_id.set(uuid.uuid4())
    try:
        context = ExecutionContext()
        event = context.record_event("retrieval", hits=3)

        assert event.elapsed_ms == 0
        assert event.stage == "retrieval"
        assert event.detail == {"hits": 3}
    finally:
        scope_id.reset(token)


def test_events_accumulate_on_the_trace_in_order() -> None:
    token = scope_id.set(uuid.uuid4())
    try:
        context = ExecutionContext()
        context.start_debug_trace()
        context.record_event("first")
        context.record_event("second")

        assert context.debug_trace.started_at > 0
        assert [e.stage for e in context.debug_trace.events] == ["first", "second"]
    finally:
        scope_id.reset(token)
