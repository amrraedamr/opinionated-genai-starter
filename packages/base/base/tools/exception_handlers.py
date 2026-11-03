import traceback as tb

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from base.models.definitions.exceptions import ExceptionCategory
from base.models.view_models import DebugContext, ErrorResponse
from base.services.common.meta import Singleton, scope_id
from base.services.configuration import get_settings
from base.services.logging.console import ConsoleLoggerService
from base.tools.errors import ErrorsException


class AppExceptionHandler(metaclass=Singleton):
    def __init__(self) -> None:
        self._logger = ConsoleLoggerService()

    @staticmethod
    def _is_debug() -> bool:
        return get_settings().environment != "production"

    @staticmethod
    def _get_request_id() -> str:
        try:
            return str(scope_id.get())
        except LookupError:
            return "unknown"

    def _build_debug_context(
        self,
        request: Request,
        traceback_lines: list[str] | None = None,
    ) -> DebugContext | None:
        if not self._is_debug():
            return None
        return DebugContext(
            request_id=self._get_request_id(),
            path=str(request.url.path),
            method=request.method,
            traceback=traceback_lines,
        )

    def _build_response(
        self,
        error: str,
        status_code: int,
        request: Request,
        traceback_lines: list[str] | None = None,
    ) -> JSONResponse:
        debug = self._build_debug_context(request, traceback_lines)
        response = ErrorResponse(error=error, debug=debug)
        return JSONResponse(
            content=response.serialize(exclude_none=True),
            status_code=status_code,
        )

    async def handle_request_validation_error(
        self, request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        field_errors: list[str] = []
        for err in errors:
            loc = " -> ".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "invalid")
            field_errors.append(f"{loc}: {msg}")

        error_message = "Validation error: " + "; ".join(field_errors)

        self._logger.warning(
            "Validation error on %s %s: %s",
            request.method,
            request.url.path,
            error_message,
        )

        return self._build_response(
            error=error_message,
            status_code=422,
            request=request,
        )

    async def handle_starlette_http_exception(
        self, request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        status_code = exc.status_code
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

        if status_code >= 500:
            self._logger.error(
                "HTTP %s on %s %s: %s",
                status_code,
                request.method,
                request.url.path,
                detail,
                status=status_code,
                category=ExceptionCategory.API,
                location=request.url.path,
            )
        else:
            self._logger.warning(
                "HTTP %s on %s %s: %s",
                status_code,
                request.method,
                request.url.path,
                detail,
            )

        return self._build_response(
            error=detail,
            status_code=status_code,
            request=request,
        )

    async def handle_errors_exception(
        self, request: Request, exc: ErrorsException
    ) -> JSONResponse:
        if exc.error_enum:
            error_message: str = exc.error_enum.value["message"]
            status_code: int = exc.error_enum.value.get("status_code", 500)
        else:
            error_message = exc.message or "Unknown Error"
            status_code = exc.status_code or 500

        if status_code >= 500:
            self._logger.error(
                "ErrorsException %s on %s %s: %s",
                status_code,
                request.method,
                request.url.path,
                error_message,
                status=status_code,
                category=ExceptionCategory.INTERNAL,
                location=request.url.path,
            )
        else:
            self._logger.warning(
                "ErrorsException %s on %s %s: %s",
                status_code,
                request.method,
                request.url.path,
                error_message,
            )

        return self._build_response(
            error=error_message,
            status_code=status_code,
            request=request,
        )

    async def handle_unhandled_exception(
        self, request: Request, exc: Exception
    ) -> JSONResponse:
        traceback_lines = tb.format_exception(type(exc), exc, exc.__traceback__)

        self._logger.exception(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
            status=500,
            category=ExceptionCategory.INTERNAL,
            location=request.url.path,
        )

        is_debug = self._is_debug()
        error_message = str(exc) if is_debug else "Internal server error."

        return self._build_response(
            error=error_message,
            status_code=500,
            request=request,
            traceback_lines=traceback_lines if is_debug else None,
        )

    def register(self, app: FastAPI) -> None:
        app.add_exception_handler(
            RequestValidationError,
            self.handle_request_validation_error,  # type: ignore
        )
        app.add_exception_handler(
            StarletteHTTPException,
            self.handle_starlette_http_exception,  # type: ignore
        )
        app.add_exception_handler(ErrorsException, self.handle_errors_exception)  # type: ignore
        app.add_exception_handler(Exception, self.handle_unhandled_exception)  # type: ignore
