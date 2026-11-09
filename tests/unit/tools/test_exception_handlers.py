"""`tools/exception_handlers.py`: what a client sees when something fails.

The envelope is the contract: a camelised `ErrorResponse`, the right status,
and a traceback only where one is safe to show. Outside `local` the message
for an unhandled error must not be the exception's own text.
"""

from collections.abc import Iterator
from json import loads
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from base.models.definitions.errors import AuthErrors
from base.services.common.meta import purge_all, scope_id
from base.tools.errors import ErrorsException
from base.tools.exception_handlers import AppExceptionHandler
from tests.conftest import MakeSettings


@pytest.fixture(autouse=True)
def _purge_instances() -> Iterator[None]:
    """The handler and the logger are singletons; settings differ per test."""
    purge_all()
    yield
    purge_all()


def request(path: str = "/widgets", method: str = "POST") -> Request:
    """A Starlette request with only what the handler reads off it."""
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
            "root_path": "",
        }
    )


def body(response: Any) -> dict[str, Any]:
    return loads(response.body)


async def test_a_validation_error_names_every_field_that_failed(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    handler = AppExceptionHandler()
    error = RequestValidationError([{"loc": ("body", "name"), "msg": "field required"}])
    response = await handler.handle_request_validation_error(request(), error)
    assert response.status_code == 422
    payload = body(response)
    assert payload["success"] is False
    assert "body -> name: field required" in payload["error"]


async def test_an_http_exception_keeps_its_own_status_and_detail(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    handler = AppExceptionHandler()
    response = await handler.handle_starlette_http_exception(
        request(), StarletteHTTPException(status_code=404, detail="Widget not found")
    )
    assert response.status_code == 404
    assert body(response)["error"] == "Widget not found"


async def test_a_server_side_http_exception_is_still_reported_to_the_client(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    handler = AppExceptionHandler()
    response = await handler.handle_starlette_http_exception(
        request(), StarletteHTTPException(status_code=502, detail="Upstream failed")
    )
    assert response.status_code == 502
    assert body(response)["error"] == "Upstream failed"


async def test_a_catalogued_error_answers_with_its_message_and_status(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    handler = AppExceptionHandler()
    response = await handler.handle_errors_exception(
        request(), ErrorsException(error_enum=AuthErrors.USER_UNAUTHORIZED)
    )
    assert response.status_code == 401
    assert body(response)["error"] == "Unauthorized."


async def test_an_ad_hoc_error_answers_with_its_own_message_and_status(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    handler = AppExceptionHandler()
    response = await handler.handle_errors_exception(
        request(), ErrorsException(message="Quota exceeded", status_code=429)
    )
    assert response.status_code == 429
    assert body(response)["error"] == "Quota exceeded"


async def test_an_error_with_neither_message_nor_status_falls_back_to_five_hundred(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    handler = AppExceptionHandler()
    response = await handler.handle_errors_exception(request(), ErrorsException())
    assert response.status_code == 500
    assert body(response)["error"] == "Unknown Error"


async def test_an_unhandled_exception_shows_its_cause_outside_production(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    handler = AppExceptionHandler()
    response = await handler.handle_unhandled_exception(
        request(), RuntimeError("connection pool exhausted")
    )
    assert response.status_code == 500
    payload = body(response)
    assert payload["error"] == "connection pool exhausted"
    assert payload["debug"]["traceback"]


async def test_an_unhandled_exception_says_nothing_specific_in_production(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="production")
    handler = AppExceptionHandler()
    response = await handler.handle_unhandled_exception(
        request(), RuntimeError("connection pool exhausted")
    )
    payload = body(response)
    assert payload["error"] == "Internal server error."
    assert "connection pool exhausted" not in response.body.decode()
    assert "debug" not in payload


async def test_the_debug_block_names_the_request_it_describes(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    handler = AppExceptionHandler()
    response = await handler.handle_starlette_http_exception(
        request("/widgets/7", "DELETE"), StarletteHTTPException(404, "gone")
    )
    debug = body(response)["debug"]
    assert debug["path"] == "/widgets/7"
    assert debug["method"] == "DELETE"


async def test_the_debug_block_carries_the_current_scope_id(
    make_settings: MakeSettings,
) -> None:
    from uuid import UUID

    make_settings(environment="local")
    scope_id.set(UUID(int=42))
    handler = AppExceptionHandler()
    response = await handler.handle_starlette_http_exception(
        request(), StarletteHTTPException(404, "gone")
    )
    assert body(response)["debug"]["requestId"] == str(UUID(int=42))


def test_a_request_outside_any_scope_reports_an_unknown_id(
    make_settings: MakeSettings,
) -> None:
    import contextvars

    make_settings(environment="local")
    fresh = contextvars.Context()
    assert fresh.run(AppExceptionHandler._get_request_id) == "unknown"


def test_a_request_inside_a_scope_reports_that_scopes_id(
    make_settings: MakeSettings,
) -> None:
    import contextvars
    from uuid import UUID

    make_settings(environment="local")

    def inside() -> str:
        scope_id.set(UUID(int=99))
        return AppExceptionHandler._get_request_id()

    assert contextvars.Context().run(inside) == str(UUID(int=99))


async def test_the_debug_block_is_withheld_in_production(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="production")
    handler = AppExceptionHandler()
    response = await handler.handle_starlette_http_exception(
        request(), StarletteHTTPException(404, "gone")
    )
    assert "debug" not in body(response)


def test_registering_installs_a_handler_for_each_failure_kind(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    app = FastAPI()
    AppExceptionHandler().register(app)
    assert RequestValidationError in app.exception_handlers
    assert StarletteHTTPException in app.exception_handlers
    assert ErrorsException in app.exception_handlers
    assert Exception in app.exception_handlers


def test_the_handler_is_one_object_for_the_whole_process(
    make_settings: MakeSettings,
) -> None:
    make_settings(environment="local")
    assert AppExceptionHandler() is AppExceptionHandler()
