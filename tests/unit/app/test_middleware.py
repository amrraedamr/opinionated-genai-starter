"""Every HTTP service shares these four pieces of middleware.

They are the floor a replica stands on: a request scope that is opened and
torn down, a correlation id that survives a hop between services, a body size
ceiling enforced twice, and per-client backpressure. Each is exercised through
a real ASGI app rather than by calling its internals.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from base.middleware import SECURITY_HEADERS, request_context
from base.middleware.request_size import RequestSizeLimitMiddleware
from base.middleware.traffic_control import (
    TrafficControlMiddleware,
)
from base.models.definitions.realtime import RealtimeCloseCode
from base.services.common.meta import Stored, scope_id
from base.tools.correlation import (
    CORRELATION_HEADER,
    TRACEPARENT_HEADER,
    current_correlation_id,
)


def context_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(request_context)

    @app.get("/echo")
    async def echo() -> dict[str, str]:
        return {
            "correlation_id": str(current_correlation_id()),
            "scope_id": str(scope_id.get()),
        }

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def body() -> Any:
            yield b"data: hello\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    return app


# request_context


def test_every_response_carries_a_correlation_id_and_a_request_id() -> None:
    with TestClient(context_app()) as client:
        response = client.get("/echo")

    assert uuid.UUID(response.headers[CORRELATION_HEADER])
    assert uuid.UUID(response.headers["X-Request-ID"])


def test_an_inbound_correlation_id_is_adopted_rather_than_replaced() -> None:
    incoming = str(uuid.uuid4())

    with TestClient(context_app()) as client:
        response = client.get("/echo", headers={CORRELATION_HEADER: incoming})

    assert response.headers[CORRELATION_HEADER] == incoming
    assert response.json()["correlation_id"] == incoming


def test_a_traceparent_supplies_the_correlation_id_when_no_header_does() -> None:
    trace_id = uuid.uuid4().hex

    with TestClient(context_app()) as client:
        response = client.get(
            "/echo",
            headers={TRACEPARENT_HEADER: f"00-{trace_id}-0123456789abcdef-01"},
        )

    assert response.headers[CORRELATION_HEADER] == str(uuid.UUID(trace_id))


def test_an_explicit_correlation_header_beats_a_traceparent() -> None:
    incoming = str(uuid.uuid4())

    with TestClient(context_app()) as client:
        response = client.get(
            "/echo",
            headers={
                CORRELATION_HEADER: incoming,
                TRACEPARENT_HEADER: f"00-{uuid.uuid4().hex}-0123456789abcdef-01",
            },
        )

    assert response.headers[CORRELATION_HEADER] == incoming


def test_an_unparseable_correlation_header_starts_a_fresh_id() -> None:
    with TestClient(context_app()) as client:
        response = client.get("/echo", headers={CORRELATION_HEADER: "not-a-uuid"})

    assert response.headers[CORRELATION_HEADER] != "not-a-uuid"
    assert uuid.UUID(response.headers[CORRELATION_HEADER])


def test_an_inbound_request_id_is_echoed_back() -> None:
    with TestClient(context_app()) as client:
        response = client.get("/echo", headers={"X-Request-ID": "req-123"})

    assert response.headers["X-Request-ID"] == "req-123"


def test_every_security_header_is_set() -> None:
    with TestClient(context_app()) as client:
        response = client.get("/echo")

    for header, value in SECURITY_HEADERS.items():
        assert response.headers[header] == value


def test_a_handler_that_set_a_security_header_keeps_its_own_value() -> None:
    app = FastAPI()
    app.middleware("http")(request_context)

    @app.get("/frame")
    async def frame() -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse({}, headers={"X-Frame-Options": "SAMEORIGIN"})

    with TestClient(app) as client:
        assert client.get("/frame").headers["X-Frame-Options"] == "SAMEORIGIN"


def test_each_request_gets_its_own_scope() -> None:
    with TestClient(context_app()) as client:
        first = client.get("/echo").json()["scope_id"]
        second = client.get("/echo").json()["scope_id"]

    assert first != second


def test_the_scope_is_torn_down_once_a_plain_response_is_finished() -> None:
    with TestClient(context_app()) as client:
        body = client.get("/echo").json()

    assert uuid.UUID(body["scope_id"]) not in Stored._scoped_instances


def test_a_streamed_response_keeps_its_scope_until_the_producer_ends_it() -> None:
    app = FastAPI()
    app.middleware("http")(request_context)
    seen: list[uuid.UUID] = []

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        from base.services.execution_context import ExecutionContext

        ExecutionContext()
        seen.append(scope_id.get())

        async def body() -> Any:
            yield b"data: hello\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    with TestClient(app) as client:
        client.get("/stream")

    assert seen[0] in Stored._scoped_instances
    Stored.delete_scope(seen[0])


# RequestSizeLimitMiddleware


def sized_app(max_bytes: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=max_bytes)

    @app.post("/upload")
    async def upload(request: Request) -> dict[str, int]:
        return {"received": len(await request.body())}

    return app


def test_a_body_within_the_limit_is_accepted() -> None:
    with TestClient(sized_app(100)) as client:
        response = client.post("/upload", content=b"x" * 50)

    assert response.status_code == 200
    assert response.json() == {"received": 50}


def test_a_declared_length_over_the_limit_is_refused_before_the_body_arrives() -> None:
    with TestClient(sized_app(10)) as client:
        response = client.post("/upload", content=b"x" * 100)

    assert response.status_code == 413
    assert response.json()["error"] == "Request body too large"


def test_a_malformed_content_length_is_a_bad_request() -> None:
    with TestClient(sized_app(100)) as client:
        response = client.post(
            "/upload", content=b"xx", headers={"Content-Length": "many"}
        )

    assert response.status_code == 400
    assert response.json()["error"] == "Invalid Content-Length"


def test_a_chunked_body_is_cut_off_once_it_passes_the_limit() -> None:
    def chunks() -> Any:
        for _ in range(10):
            yield b"x" * 20

    with TestClient(sized_app(50)) as client:
        response = client.post("/upload", content=chunks())

    assert response.status_code == 413


def test_a_non_http_scope_is_passed_straight_through() -> None:
    seen: list[str] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    import asyncio

    middleware = RequestSizeLimitMiddleware(inner, max_bytes=1)
    asyncio.run(middleware({"type": "lifespan"}, None, None))  # type: ignore[arg-type]

    assert seen == ["lifespan"]


# TrafficControlMiddleware


def budgets(
    per_minute: int = 100, auth_per_minute: int = 100, ws_connect: int = 100
) -> Any:
    """A limiter with budgets this test chose, not whatever settings hold."""
    from dataclasses import dataclass

    from base.models.definitions.rate_limiting import RateLimitBackend
    from base.services.rate_limiting.memory import InMemoryRateLimiter

    @dataclass
    class Budgets:
        rate_limit_backend: RateLimitBackend = RateLimitBackend.MEMORY
        rate_limit_requests_per_minute: int = per_minute
        rate_limit_auth_per_minute: int = auth_per_minute
        rate_limit_llm_tokens_per_minute: int = 100_000
        rate_limit_ws_connect_per_minute: int = ws_connect
        rate_limit_ws_messages_per_minute: int = 100

    return InMemoryRateLimiter(Budgets())


def limited_app(
    per_minute: int = 100,
    auth_per_minute: int = 100,
    concurrency: int = 100,
    ws_connect: int = 100,
) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        TrafficControlMiddleware,
        concurrency=concurrency,
        limiter=budgets(per_minute, auth_per_minute, ws_connect),
    )

    @app.get("/thing")
    async def thing() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/auth/login")
    async def login() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/auth")
    async def api_auth() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws")
    async def socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    return app


def test_requests_under_the_rate_limit_are_served() -> None:
    with TestClient(limited_app(per_minute=3)) as client:
        assert [client.get("/thing").status_code for _ in range(3)] == [200, 200, 200]


def test_the_request_after_the_limit_is_refused_with_a_retry_hint() -> None:
    with TestClient(limited_app(per_minute=2)) as client:
        client.get("/thing")
        client.get("/thing")
        response = client.get("/thing")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"
    assert response.json()["error"] == "Too many requests"


def test_auth_endpoints_get_their_own_stricter_budget() -> None:
    with TestClient(limited_app(per_minute=100, auth_per_minute=1)) as client:
        assert client.post("/auth/login").status_code == 200
        assert client.post("/auth/login").status_code == 429
        assert client.get("/thing").status_code == 200


def test_the_auth_budget_does_not_consume_the_general_one() -> None:
    with TestClient(limited_app(per_minute=1, auth_per_minute=5)) as client:
        assert client.post("/auth/login").status_code == 200
        assert client.get("/thing").status_code == 200


def test_a_request_that_cannot_get_a_concurrency_slot_is_told_the_server_is_busy() -> (
    None
):
    app = limited_app(concurrency=1)
    middleware = next(
        m for m in app.user_middleware if m.cls is TrafficControlMiddleware
    )
    assert middleware.kwargs["concurrency"] == 1

    with TestClient(app) as client:
        built = client.app.middleware_stack  # type: ignore[attr-defined]
        traffic = _find_traffic_control(built)
        assert traffic is not None
        traffic._semaphore._value = 0

        response = client.get("/thing")

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json()["error"] == "Server is busy"


def test_a_path_ending_in_auth_also_gets_the_auth_budget() -> None:
    with TestClient(limited_app(per_minute=100, auth_per_minute=1)) as client:
        assert client.get("/api/auth").status_code == 200
        assert client.get("/api/auth").status_code == 429


# Websockets


def test_a_websocket_under_the_budget_is_admitted() -> None:
    with TestClient(limited_app(ws_connect=2)) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_text() == "hello"


def test_a_websocket_over_the_budget_is_refused_at_the_handshake() -> None:
    """Sockets bypassed this middleware entirely before, so nothing bounded them."""
    with TestClient(limited_app(ws_connect=1)) as client:
        with client.websocket_connect("/ws") as socket:
            socket.receive_text()

        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/ws"):
                pass

    assert refused.value.code == RealtimeCloseCode.TRY_LATER


def test_a_handshake_does_not_spend_the_http_budget() -> None:
    """A connection and a request are different budgets, counted separately."""
    with TestClient(limited_app(per_minute=1, ws_connect=5)) as client:
        with client.websocket_connect("/ws") as socket:
            socket.receive_text()
        assert client.get("/thing").status_code == 200


def test_a_socket_does_not_hold_a_concurrency_slot() -> None:
    """A slot held for the life of a socket would exhaust the ceiling."""
    app = limited_app(concurrency=1, ws_connect=5)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.receive_text()
        traffic = _find_traffic_control(
            client.app.middleware_stack  # type: ignore[attr-defined]
        )
        assert traffic is not None
        assert traffic._semaphore._value == 1


def test_a_lifespan_message_passes_straight_through() -> None:
    """Startup is not traffic, and must not be spent against a budget."""
    with TestClient(limited_app(per_minute=1)) as client:
        assert client.get("/thing").status_code == 200


def _find_traffic_control(app: Any) -> TrafficControlMiddleware | None:
    """Walk the built ASGI stack for the live middleware instance."""
    seen = 0
    while app is not None and seen < 20:
        if isinstance(app, TrafficControlMiddleware):
            return app
        app = getattr(app, "app", None)
        seen += 1
    return None
