import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from types import ModuleType

from base.tools.errors import MissingExtraError
from base.tools.misc import split_csv

try:
    from fastapi import FastAPI, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("server", "fastapi", feature="create_app") from exc

from base.capabilities import Capability
from base.middleware import (
    RequestSizeLimitMiddleware,
    TrafficControlMiddleware,
    request_context,
)
from base.models.definitions.health import ProbeStatus
from base.models.view_models.health import HealthReport, ProbeResult
from base.services.configuration.protocols import SupportsServerConfig
from base.services.logging.console import ConsoleLoggerService
from base.tools.exception_handlers import AppExceptionHandler
from base.tools.routers import init_routers

LifespanHook = Callable[[FastAPI], AbstractAsyncContextManager[None]]
DEFAULT_TITLE = "Opinionated GenAI Starter API"
DEFAULT_VERSION = "0.1"


def _resolve_cors_origins(
    cors_origins: Sequence[str] | None,
    environment: str,
    logger: ConsoleLoggerService,
) -> list[str]:
    if cors_origins is not None:
        return list(cors_origins)
    if environment == "local":
        return ["*"]
    logger.warning("No CORS origins configured; cross-origin requests are refused")
    return []


async def _run_probe(capability: Capability) -> ProbeResult:
    try:
        return await capability.aprobe()
    except Exception as exc:
        return ProbeResult(
            name=capability.name,
            status=ProbeStatus.FAILED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def create_app(
    *,
    settings: SupportsServerConfig,
    routers: ModuleType | None = None,
    prefix: str = "/api",
    capabilities: Sequence[Capability] = (),
    lifespan_hooks: Sequence[LifespanHook] = (),
    cors_origins: Sequence[str] | None = None,
    title: str = DEFAULT_TITLE,
    version: str = DEFAULT_VERSION,
) -> FastAPI:
    """Build one service's FastAPI application.

    `settings` must satisfy `SupportsServerConfig`, so a service with no HTTP
    surface cannot reach here.
    """
    logger = ConsoleLoggerService()
    capabilities = tuple(capabilities)
    lifespan_hooks = tuple(lifespan_hooks)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        started: list[Capability] = []
        try:
            async with AsyncExitStack() as stack:
                for hook in lifespan_hooks:
                    await stack.enter_async_context(hook(app))
                for capability in capabilities:
                    await capability.astartup(app)
                    started.append(capability)
                yield
        finally:
            for capability in reversed(started):
                try:
                    await capability.ashutdown(app)
                except Exception:
                    logger.exception(
                        "Capability failed to shut down: %s", capability.name
                    )

    app = FastAPI(title=title, version=version, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_cors_origins(cors_origins, settings.environment, logger),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    trusted_hosts = split_csv(settings.trusted_hosts)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts or ["*"])
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.request_max_bytes)
    app.add_middleware(
        TrafficControlMiddleware,
        concurrency=settings.request_concurrency_limit,
        per_minute=settings.request_rate_limit_per_minute,
        auth_per_minute=settings.auth_rate_limit_per_minute,
    )
    app.middleware("http")(request_context)

    if routers is not None:
        init_routers(app, routers=routers, prefix=prefix)
    AppExceptionHandler().register(app)

    @app.get("/", tags=["Main"])
    def root() -> dict[str, str]:
        return {"message": f"Welcome to the {title}!", "version": version}

    @app.get("/health-check", tags=["Main"])
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/livez", tags=["Main"])
    async def livez() -> HealthReport:
        return HealthReport(status=ProbeStatus.OK)

    @app.get("/readyz", tags=["Main"])
    async def readyz() -> Response:
        checks = list(await asyncio.gather(*(_run_probe(c) for c in capabilities)))
        report = HealthReport.aggregate(checks)
        return JSONResponse(
            content=report.serialize(exclude_none=True),
            status_code=503 if report.status is ProbeStatus.FAILED else 200,
        )

    return app
