"""Every route on the conversation surface, through the application itself.

The guard and the ownership scope are the point: authentication is a
dependency rather than a check inside a handler, and a conversation is only
ever reachable through a query that already names its owner. Both are read
off real HTTP responses, with genuine signed tokens, so a route that forgot
either would fail here rather than in production.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from base.models.definitions.canned_messages import CannedMessages
from base.models.definitions.database import Locale
from base.models.definitions.errors import DatabaseErrors
from base.models.view_models.flow import FlowAbortError
from base.services.auth import TokenIssuer, current_user_id
from base.services.common.meta import Stored
from base.services.execution_context import ExecutionContext
from base.services.sql.service import SQLService
from base.tools.database import SQLResultFormats
from base.tools.errors import ErrorsException
from chat_api.models.orm.chat_history_message import ChatHistoryMessageDTO
from chat_api.routers import conversation as conversation_router
from chat_api.stores.conversation import ConversationStore

from .conftest import StubAgent, access_token, conversation_of

ARABIC = "مرحبا بالعالم"
EMOJI = "🙂 zero-width:\u200b combining: é"


@pytest.fixture
def conversation(client: TestClient, auth: dict[str, str]) -> str:
    return conversation_of(client, auth, title="A chat")


# The guard


def test_a_request_with_no_token_is_refused(client: TestClient) -> None:
    assert client.post("/api/conversations", json={}).status_code == 401


def test_a_request_with_a_token_of_the_wrong_scheme_is_refused(
    client: TestClient, user_id: uuid.UUID
) -> None:
    response = client.post(
        "/api/conversations",
        json={},
        headers={"Authorization": f"Basic {access_token(user_id)}"},
    )
    assert response.status_code == 401


def test_a_token_that_is_not_a_token_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/conversations", json={}, headers={"Authorization": "Bearer nonsense"}
    )
    assert response.status_code == 401


def test_a_refresh_token_does_not_open_an_access_route(
    client: TestClient, user_id: uuid.UUID
) -> None:
    refresh = TokenIssuer().issue_refresh({"user_id": str(user_id)})
    response = client.post(
        "/api/conversations", json={}, headers={"Authorization": f"Bearer {refresh}"}
    )
    assert response.status_code == 401


def test_a_token_whose_subject_is_not_a_user_id_is_refused(
    client: TestClient,
) -> None:
    """An IdP subject is rarely a UUID, so `Principal.user_id` can be absent.

    Scoping a query to `user_id IS NULL` would hide every row from such a
    caller on reads and break the NOT NULL constraint on a write.
    """
    token = TokenIssuer().issue_access({"sub": "alice@example.com"})
    response = client.post(
        "/api/conversations", json={}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401
    assert response.json()["error"] == "Token incomplete."


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/conversations"),
        ("get", "/api/conversations/{id}"),
        ("patch", "/api/conversations/{id}"),
        ("delete", "/api/conversations/{id}"),
        ("post", "/api/conversations/{id}/messages"),
        ("get", "/api/conversations/{id}/messages"),
    ],
)
def test_every_route_is_behind_the_guard(
    client: TestClient, method: str, path: str
) -> None:
    response = client.request(method, path.format(id=uuid.uuid4()), json={})
    assert response.status_code == 401


# Create


def test_creating_a_conversation_returns_its_id(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post("/api/conversations", json={"title": "A"}, headers=auth)

    assert response.status_code == 200
    assert response.json()["success"] is True
    uuid.UUID(response.json()["data"]["conversationId"])


def test_a_conversation_created_without_a_title_gets_the_default(
    client: TestClient, auth: dict[str, str]
) -> None:
    created = conversation_of(client, auth)
    body = client.get(f"/api/conversations/{created}", headers=auth).json()
    assert body["data"]["title"] == "New Chat"


def test_metadata_survives_the_round_trip(
    client: TestClient, auth: dict[str, str]
) -> None:
    metadata = {"source": "web", "nested": {"a": [1, 2]}, "unicode": ARABIC}
    created = conversation_of(client, auth, title="t", metadata=metadata)

    body = client.get(f"/api/conversations/{created}", headers=auth).json()
    assert body["data"]["metadata"] == metadata


def test_a_title_may_be_awkward_unicode(
    client: TestClient, auth: dict[str, str]
) -> None:
    created = conversation_of(client, auth, title=EMOJI)
    body = client.get(f"/api/conversations/{created}", headers=auth).json()
    assert body["data"]["title"] == EMOJI


def test_a_conversation_is_created_for_the_caller_the_token_names(
    client: TestClient, auth: dict[str, str], user_id: uuid.UUID
) -> None:
    created = conversation_of(client, auth)
    stored = ConversationStore().get_for_user(user_id, uuid.UUID(created))
    assert stored is not None


def test_a_caller_cannot_name_somebody_else_as_the_owner(
    client: TestClient,
    auth: dict[str, str],
    user_id: uuid.UUID,
    other_user_id: uuid.UUID,
) -> None:
    created = conversation_of(client, auth, title="t", userId=str(other_user_id))

    assert ConversationStore().get_for_user(other_user_id, uuid.UUID(created)) is None
    assert ConversationStore().get_for_user(user_id, uuid.UUID(created)) is not None


# Get


def test_a_conversation_reads_back_with_its_timestamps(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    body = client.get(f"/api/conversations/{conversation}", headers=auth).json()["data"]

    assert body["conversationId"] == conversation
    assert body["title"] == "A chat"
    assert body["createdAt"] and body["updatedAt"]


def test_a_conversation_that_does_not_exist_is_not_found(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.get(f"/api/conversations/{uuid.uuid4()}", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"] == "Record not found."


def test_a_conversation_belonging_to_another_user_is_refused(
    client: TestClient, conversation: str, other_auth: dict[str, str]
) -> None:
    """404 rather than 403: a stranger learns nothing about what exists."""
    response = client.get(f"/api/conversations/{conversation}", headers=other_auth)
    assert response.status_code == 404


def test_an_identifier_that_is_not_a_uuid_is_rejected_before_the_store(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.get("/api/conversations/not-a-uuid", headers=auth)
    assert response.status_code == 422


# Edit


def test_editing_a_title_writes_it(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    response = client.patch(
        f"/api/conversations/{conversation}", json={"title": "Renamed"}, headers=auth
    )

    assert response.status_code == 200
    body = client.get(f"/api/conversations/{conversation}", headers=auth).json()
    assert body["data"]["title"] == "Renamed"


def test_editing_metadata_replaces_it_wholesale(
    client: TestClient, auth: dict[str, str]
) -> None:
    created = conversation_of(client, auth, title="t", metadata={"a": 1})
    client.patch(
        f"/api/conversations/{created}", json={"metadata": {"b": 2}}, headers=auth
    )

    body = client.get(f"/api/conversations/{created}", headers=auth).json()
    assert body["data"]["metadata"] == {"b": 2}


def test_an_empty_edit_changes_nothing(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    response = client.patch(f"/api/conversations/{conversation}", json={}, headers=auth)

    assert response.status_code == 200
    body = client.get(f"/api/conversations/{conversation}", headers=auth).json()
    assert body["data"]["title"] == "A chat"


def test_an_explicit_null_is_not_taken_as_a_change(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    """`exclude_none` means a null clears nothing — it is read as "unset"."""
    client.patch(
        f"/api/conversations/{conversation}", json={"title": None}, headers=auth
    )

    body = client.get(f"/api/conversations/{conversation}", headers=auth).json()
    assert body["data"]["title"] == "A chat"


def test_an_edit_cannot_hand_the_conversation_to_another_user(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    user_id: uuid.UUID,
    other_user_id: uuid.UUID,
) -> None:
    client.patch(
        f"/api/conversations/{conversation}",
        json={"title": "t", "userId": str(other_user_id)},
        headers=auth,
    )

    assert ConversationStore().get_for_user(user_id, uuid.UUID(conversation))
    assert (
        ConversationStore().get_for_user(other_user_id, uuid.UUID(conversation)) is None
    )


def test_editing_a_conversation_that_does_not_exist_is_not_found(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.patch(
        f"/api/conversations/{uuid.uuid4()}", json={"title": "x"}, headers=auth
    )
    assert response.status_code == 404


def test_editing_another_users_conversation_is_refused(
    client: TestClient, conversation: str, other_auth: dict[str, str]
) -> None:
    response = client.patch(
        f"/api/conversations/{conversation}", json={"title": "x"}, headers=other_auth
    )
    assert response.status_code == 404


# Delete


def test_deleting_a_conversation_takes_it_out_of_reach(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    assert (
        client.delete(f"/api/conversations/{conversation}", headers=auth).status_code
        == 200
    )
    assert (
        client.get(f"/api/conversations/{conversation}", headers=auth).status_code
        == 404
    )


def test_deleting_twice_is_not_found_the_second_time(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    client.delete(f"/api/conversations/{conversation}", headers=auth)
    assert (
        client.delete(f"/api/conversations/{conversation}", headers=auth).status_code
        == 404
    )


def test_deleting_a_conversation_that_does_not_exist_is_not_found(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.delete(f"/api/conversations/{uuid.uuid4()}", headers=auth)
    assert response.status_code == 404


def test_another_user_cannot_delete_a_conversation(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    other_auth: dict[str, str],
) -> None:
    assert (
        client.delete(
            f"/api/conversations/{conversation}", headers=other_auth
        ).status_code
        == 404
    )
    assert (
        client.get(f"/api/conversations/{conversation}", headers=auth).status_code
        == 200
    )


def test_deleting_a_conversation_takes_its_messages_with_it(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )
    client.delete(f"/api/conversations/{conversation}", headers=auth)

    rows = SQLService().query(
        select(ChatHistoryMessageDTO).where(
            ChatHistoryMessageDTO.conversation_id == uuid.UUID(conversation)
        ),
        SQLResultFormats.all_scalar_list,
    )
    assert rows and all(row.deleted_at is not None for row in rows)


# Sending a message


def test_sending_a_message_answers_with_what_the_agent_produced(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    response = client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )

    assert response.status_code == 200
    assert response.json()["data"]["content"] == "the answer"


def test_sending_a_message_stores_both_turns_in_order(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )

    messages = client.get(
        f"/api/conversations/{conversation}/messages", headers=auth
    ).json()["data"]["messages"]
    assert [(m["role"], m["content"], m["sequence"]) for m in messages] == [
        ("user", "hello", 1),
        ("assistant", "the answer", 2),
    ]


def test_a_second_turn_continues_the_sequence(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    for text in ("first", "second"):
        client.post(
            f"/api/conversations/{conversation}/messages",
            json={"content": text},
            headers=auth,
        )

    messages = client.get(
        f"/api/conversations/{conversation}/messages", headers=auth
    ).json()["data"]["messages"]
    assert [m["sequence"] for m in messages] == [1, 2, 3, 4]


def test_the_debug_trace_is_withheld_unless_it_is_asked_for(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    response = client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )
    assert response.json()["data"]["debug"] is None


def test_asking_for_debug_returns_the_trace(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    response = client.post(
        f"/api/conversations/{conversation}/messages?debug=true",
        json={"content": "hello"},
        headers=auth,
    )
    assert response.json()["data"]["debug"] is not None


def test_awkward_unicode_survives_a_turn(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": EMOJI},
        headers=auth,
    )

    messages = client.get(
        f"/api/conversations/{conversation}/messages", headers=auth
    ).json()["data"]["messages"]
    assert messages[0]["content"] == EMOJI


def test_an_arabic_question_is_answered_in_its_own_locale(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    """The locale the flow runs under is read off the question, not a header."""
    seen: list[Locale] = []

    async def arun(self: Any) -> None:
        seen.append(ExecutionContext().flow_variables.locale)
        ExecutionContext().flow_variables.final_answer = "ok"

    agent.arun = arun  # type: ignore[method-assign]
    client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": ARABIC},
        headers=auth,
    )

    assert seen == [Locale.AR]


def test_an_empty_message_is_rejected(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    response = client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": ""},
        headers=auth,
    )
    assert response.status_code == 422
    assert "at least 1 character" in response.json()["error"]


def test_a_message_longer_than_the_limit_is_rejected(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    response = client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "x" * 16_001},
        headers=auth,
    )
    assert response.status_code == 422


def test_a_message_exactly_at_the_limit_is_accepted(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    response = client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "x" * 16_000},
        headers=auth,
    )
    assert response.status_code == 200


def test_a_message_with_no_content_field_is_rejected(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    response = client.post(
        f"/api/conversations/{conversation}/messages", json={}, headers=auth
    )
    assert response.status_code == 422


def test_sending_to_a_conversation_that_does_not_exist_is_not_found(
    client: TestClient, auth: dict[str, str], agent: type[StubAgent]
) -> None:
    response = client.post(
        f"/api/conversations/{uuid.uuid4()}/messages",
        json={"content": "hello"},
        headers=auth,
    )
    assert response.status_code == 404


def test_sending_to_another_users_conversation_is_refused(
    client: TestClient,
    conversation: str,
    other_auth: dict[str, str],
    agent: type[StubAgent],
) -> None:
    response = client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=other_auth,
    )
    assert response.status_code == 404


def test_nothing_is_written_when_the_conversation_is_not_the_callers(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    other_auth: dict[str, str],
    agent: type[StubAgent],
) -> None:
    client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=other_auth,
    )

    messages = client.get(
        f"/api/conversations/{conversation}/messages", headers=auth
    ).json()["data"]["messages"]
    assert messages == []


def test_an_aborted_flow_answers_with_the_canned_message(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    agent.failure = FlowAbortError(CannedMessages.LLM_GENERIC_ERROR)
    response = client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )

    assert response.status_code == 200
    assert (
        response.json()["data"]["content"]
        == (CannedMessages.LLM_GENERIC_ERROR.value[Locale.EN])
    )


def test_a_failing_flow_leaves_the_user_turn_behind(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    """The question is appended before the agent runs, so it survives a failure."""
    agent.failure = FlowAbortError(CannedMessages.LLM_GENERIC_ERROR)
    client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )

    messages = client.get(
        f"/api/conversations/{conversation}/messages", headers=auth
    ).json()["data"]["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [("user", "hello")]


def test_an_error_raised_outside_a_stream_reaches_the_caller(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    agent.failure = ErrorsException(error_enum=DatabaseErrors.RECORD_NOT_FOUND)
    response = client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )
    assert response.status_code == 404


def test_an_unexpected_failure_outside_a_stream_is_not_swallowed(
    app: Any, auth: dict[str, str], conversation: str, agent: type[StubAgent]
) -> None:
    """There is no error frame to send, so the exception has to propagate."""
    agent.failure = RuntimeError("the agent exploded")
    quiet = TestClient(app, raise_server_exceptions=False)

    response = quiet.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )
    assert response.status_code == 500


# Streaming


def read_stream(
    client: TestClient, path: str, headers: dict[str, str], body: dict[str, Any]
) -> list[str]:
    """Collect an SSE response's lines; the agent closes the stream itself."""
    with client.stream("POST", path, json=body, headers=headers) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return [line for line in response.iter_lines()]


def test_a_streaming_turn_is_served_as_an_event_stream(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    lines = read_stream(
        client,
        f"/api/conversations/{conversation}/messages?streaming=true",
        auth,
        {"content": "hello"},
    )

    assert any('"data": "the answer"' in line for line in lines)
    assert any('"message_type": "end"' in line for line in lines)


def test_a_streaming_turn_still_stores_both_sides(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    """The answer is appended by the background task, after the stream closed."""
    read_stream(
        client,
        f"/api/conversations/{conversation}/messages?streaming=true",
        auth,
        {"content": "hello"},
    )

    deadline = time.monotonic() + 5
    while True:
        messages = client.get(
            f"/api/conversations/{conversation}/messages", headers=auth
        ).json()["data"]["messages"]
        if len(messages) == 2 or time.monotonic() > deadline:
            break
        time.sleep(0.01)
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_a_stream_that_fails_sends_an_error_frame_rather_than_hanging(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    agent.failure = ErrorsException(error_enum=DatabaseErrors.RECORD_NOT_FOUND)
    lines = read_stream(
        client,
        f"/api/conversations/{conversation}/messages?streaming=true",
        auth,
        {"content": "hello"},
    )

    assert any('"message_type": "error"' in line for line in lines)


def test_an_unexpected_failure_in_a_stream_says_nothing_about_itself(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    agent.failure = RuntimeError("the password is hunter2")
    lines = read_stream(
        client,
        f"/api/conversations/{conversation}/messages?streaming=true",
        auth,
        {"content": "hello"},
    )

    assert any("Internal server error" in line for line in lines)
    assert not any("hunter2" in line for line in lines)


# Listing messages


def test_a_fresh_conversation_has_no_messages(
    client: TestClient, auth: dict[str, str], conversation: str
) -> None:
    response = client.get(f"/api/conversations/{conversation}/messages", headers=auth)

    assert response.status_code == 200
    assert response.json()["data"]["messages"] == []


def test_listing_messages_of_a_conversation_that_does_not_exist_is_not_found(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.get(f"/api/conversations/{uuid.uuid4()}/messages", headers=auth)
    assert response.status_code == 404


def test_listing_another_users_messages_is_refused(
    client: TestClient, conversation: str, other_auth: dict[str, str]
) -> None:
    response = client.get(
        f"/api/conversations/{conversation}/messages", headers=other_auth
    )
    assert response.status_code == 404


def test_one_conversations_messages_never_appear_in_another(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    other = conversation_of(client, auth, title="Other")
    client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )

    messages = client.get(f"/api/conversations/{other}/messages", headers=auth).json()
    assert messages["data"]["messages"] == []


def test_the_request_scope_does_not_leak_between_turns(
    client: TestClient,
    auth: dict[str, str],
    conversation: str,
    agent: type[StubAgent],
) -> None:
    """Each turn builds its own `ExecutionContext`; the previous one is torn down."""
    client.post(
        f"/api/conversations/{conversation}/messages",
        json={"content": "hello"},
        headers=auth,
    )
    assert Stored._scoped_instances == {}


def test_the_dependency_is_shared_rather_than_repeated_per_route() -> None:
    """One object, so a route cannot acquire its own looser copy of the guard."""
    dependencies = {
        tuple(d.call for d in route.dependant.dependencies)  # type: ignore[attr-defined]
        for route in conversation_router.router.routes
    }
    assert dependencies == {(current_user_id,)}
