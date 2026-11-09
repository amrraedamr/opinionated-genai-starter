"""`create_app` is where one service becomes an application.

It is the only place routers, middleware, capabilities and the health surface
are wired together, so the order matters and nothing else gets to decide it.
Capabilities start inside the lifespan and shut down in reverse, which is the
only way a dependency built on another can be torn down safely.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.testclient import TestClient

from base.app import DEFAULT_TITLE, DEFAULT_VERSION, create_app
from base.middleware import RequestSizeLimitMiddleware, TrafficControlMiddleware
from base.models.definitions.health import ProbeStatus
from base.models.view_models.health import ProbeResult
from base.services.common.meta import Singleton
from base.services.configuration.mixins import ServerSettingsMixin
from base.tools.errors import MissingExtraError


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    Singleton.purge()
    yield
    Singleton.purge()


@pytest.fixture
def server_settings(make_settings: Any) -> Any:
    return make_settings(ServerSettingsMixin, service_name="demo")


class Recording:
    """A capability that writes down its own lifecycle."""

    def __init__(
        self,
        name: str,
        log: list[str],
        status: ProbeStatus = ProbeStatus.OK,
        fail_shutdown: bool = False,
    ) -> None:
        self.name = name
        self._log = log
        self._status = status
        self._fail_shutdown = fail_shutdown

    async def astartup(self, app: FastAPI) -> None:
        self._log.append(f"start:{self.name}")

    async def ashutdown(self, app: FastAPI) -> None:
        if self._fail_shutdown:
            raise RuntimeError("could not close")
        self._log.append(f"stop:{self.name}")

    async def aprobe(self) -> ProbeResult:
        return ProbeResult(name=self.name, status=self._status)


class Exploding(Recording):
    async def aprobe(self) -> ProbeResult:
        raise RuntimeError("probe blew up")


def routers_module(*paths: str) -> types.ModuleType:
    """A throwaway `routers` package with one module per path."""
    from fastapi import APIRouter

    package = types.ModuleType("scratch_routers")
    package.__path__ = []  # type: ignore[attr-defined]
    modules: list[types.ModuleType] = []
    for index, path in enumerate(paths):
        module = types.ModuleType(f"scratch_routers.m{index}")
        router = APIRouter(tags=["scratch"])

        @router.get(path)
        async def handler() -> dict[str, str]:
            return {"path": path}

        module.router = router  # type: ignore[attr-defined]
        modules.append(module)
        sys.modules[module.__name__] = module

    def iter_modules(_: Any = None) -> Any:
        return [(None, m.__name__.rsplit(".", 1)[1], False) for m in modules]

    package._iter = iter_modules  # type: ignore[attr-defined]
    return package


# Shape


def test_the_application_is_titled_and_versioned(server_settings: Any) -> None:
    app = create_app(settings=server_settings)

    assert app.title == DEFAULT_TITLE
    assert app.version == DEFAULT_VERSION


def test_a_service_may_name_its_own_title_and_version(server_settings: Any) -> None:
    app = create_app(settings=server_settings, title="Chat API", version="2.1")

    assert (app.title, app.version) == ("Chat API", "2.1")


def test_the_built_in_routes_are_always_present(server_settings: Any) -> None:
    app = create_app(settings=server_settings)

    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert {"/", "/health-check", "/livez", "/readyz"} <= paths


def test_the_root_route_names_the_service(server_settings: Any) -> None:
    with TestClient(create_app(settings=server_settings, title="Chat API")) as client:
        body = client.get("/").json()

    assert body == {"message": "Welcome to the Chat API!", "version": DEFAULT_VERSION}


def test_health_check_answers_without_touching_a_capability(
    server_settings: Any,
) -> None:
    log: list[str] = []
    app = create_app(settings=server_settings, capabilities=[Exploding("broken", log)])

    with TestClient(app) as client:
        assert client.get("/health-check").json() == {"status": "ok"}


def test_liveness_does_not_depend_on_readiness(server_settings: Any) -> None:
    log: list[str] = []
    app = create_app(
        settings=server_settings,
        capabilities=[Recording("db", log, status=ProbeStatus.FAILED)],
    )

    with TestClient(app) as client:
        assert client.get("/livez").json()["status"] == "ok"


# Middleware


def test_the_middleware_stack_is_built_in_one_fixed_order(
    server_settings: Any,
) -> None:
    app = create_app(settings=server_settings)

    # Starlette inserts each addition at the front, so index 0 is outermost.
    classes = [m.cls.__name__ for m in app.user_middleware]
    assert classes == [
        "BaseHTTPMiddleware",
        "TrafficControlMiddleware",
        "RequestSizeLimitMiddleware",
        "TrustedHostMiddleware",
        "CORSMiddleware",
    ]


def test_the_request_context_middleware_runs_outermost(server_settings: Any) -> None:
    app = create_app(settings=server_settings)

    assert len(app.user_middleware) == 5
    assert app.user_middleware[0].cls.__name__ == "BaseHTTPMiddleware"
    assert app.user_middleware[0].kwargs["dispatch"].__name__ == "request_context"


def test_traffic_control_sits_outside_the_body_size_limit(
    server_settings: Any,
) -> None:
    classes = [m.cls for m in create_app(settings=server_settings).user_middleware]

    assert classes.index(TrafficControlMiddleware) < classes.index(
        RequestSizeLimitMiddleware
    )


def test_the_size_limit_and_traffic_limits_come_from_settings(
    make_settings: Any,
) -> None:
    """Budgets no longer come through the middleware: the limiter reads them.

    Only the concurrency ceiling is passed, because it is this process's own
    and not a figure any other replica shares.
    """
    settings = make_settings(
        ServerSettingsMixin,
        service_name="demo",
        request_max_bytes=2048,
        request_concurrency_limit=7,
    )
    app = create_app(settings=settings)

    by_class = {m.cls: m for m in app.user_middleware}
    assert by_class[RequestSizeLimitMiddleware].kwargs["max_bytes"] == 2048
    assert by_class[TrafficControlMiddleware].kwargs == {"concurrency": 7}


def test_trusted_hosts_are_split_from_the_comma_separated_setting(
    make_settings: Any,
) -> None:
    settings = make_settings(
        ServerSettingsMixin,
        service_name="demo",
        trusted_hosts="api.example.com, admin.example.com",
    )
    app = create_app(settings=settings)

    by_class = {m.cls: m for m in app.user_middleware}
    assert by_class[TrustedHostMiddleware].kwargs["allowed_hosts"] == [
        "api.example.com",
        "admin.example.com",
    ]


def test_no_trusted_hosts_means_every_host_rather_than_none(
    make_settings: Any,
) -> None:
    settings = make_settings(ServerSettingsMixin, service_name="demo", trusted_hosts="")
    app = create_app(settings=settings)

    by_class = {m.cls: m for m in app.user_middleware}
    assert by_class[TrustedHostMiddleware].kwargs["allowed_hosts"] == ["*"]


def test_local_development_allows_any_origin(make_settings: Any) -> None:
    settings = make_settings(
        ServerSettingsMixin, service_name="demo", environment="local"
    )
    app = create_app(settings=settings)

    by_class = {m.cls: m for m in app.user_middleware}
    assert by_class[CORSMiddleware].kwargs["allow_origins"] == ["*"]


def test_a_deployed_service_with_no_origins_refuses_cross_origin_requests(
    make_settings: Any,
) -> None:
    settings = make_settings(
        ServerSettingsMixin, service_name="demo", environment="production"
    )
    app = create_app(settings=settings)

    by_class = {m.cls: m for m in app.user_middleware}
    assert by_class[CORSMiddleware].kwargs["allow_origins"] == []


def test_explicit_origins_are_used_as_given(make_settings: Any) -> None:
    settings = make_settings(
        ServerSettingsMixin, service_name="demo", environment="production"
    )
    app = create_app(settings=settings, cors_origins=["https://app.example.com"])

    by_class = {m.cls: m for m in app.user_middleware}
    assert by_class[CORSMiddleware].kwargs["allow_origins"] == [
        "https://app.example.com"
    ]


# Routers


def test_routers_are_registered_under_the_prefix(
    server_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pkgutil

    package = routers_module("/conversations", "/logs")
    monkeypatch.setattr(pkgutil, "iter_modules", package._iter)

    app = create_app(settings=server_settings, routers=package, prefix="/api/chat")

    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/chat/conversations" in paths
    assert "/api/chat/logs" in paths


def test_a_service_with_no_routers_still_builds(server_settings: Any) -> None:
    app = create_app(settings=server_settings, routers=None)

    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/livez" in paths


# Lifespan


def test_capabilities_start_in_order_and_stop_in_reverse(
    server_settings: Any,
) -> None:
    log: list[str] = []
    app = create_app(
        settings=server_settings,
        capabilities=[Recording("db", log), Recording("broker", log)],
    )

    with TestClient(app):
        pass

    assert log == ["start:db", "start:broker", "stop:broker", "stop:db"]


def test_lifespan_hooks_run_before_any_capability_starts_and_unwind_after(
    server_settings: Any,
) -> None:
    log: list[str] = []

    @asynccontextmanager
    async def hook(app: FastAPI) -> AsyncGenerator[None, None]:
        log.append("hook:enter")
        yield
        log.append("hook:exit")

    app = create_app(
        settings=server_settings,
        capabilities=[Recording("db", log)],
        lifespan_hooks=[hook],
    )

    with TestClient(app):
        pass

    # Strict LIFO: a hook entered before a capability started unwinds after it
    # stopped, so a capability still has whatever the hook was holding open.
    assert log == ["hook:enter", "start:db", "stop:db", "hook:exit"]


def test_a_capability_that_fails_to_start_stops_the_ones_already_started(
    server_settings: Any,
) -> None:
    log: list[str] = []

    class Refusing(Recording):
        async def astartup(self, app: FastAPI) -> None:
            raise RuntimeError("cannot connect")

    app = create_app(
        settings=server_settings,
        capabilities=[Recording("db", log), Refusing("broker", log)],
    )

    with pytest.raises(RuntimeError, match="cannot connect"), TestClient(app):
        pass

    assert log == ["start:db", "stop:db"]


def test_a_capability_that_fails_to_stop_does_not_block_the_others(
    server_settings: Any,
) -> None:
    log: list[str] = []
    app = create_app(
        settings=server_settings,
        capabilities=[
            Recording("db", log),
            Recording("broker", log, fail_shutdown=True),
        ],
    )

    with TestClient(app):
        pass

    assert log == ["start:db", "start:broker", "stop:db"]


# Readiness


def test_readiness_reports_every_capability(server_settings: Any) -> None:
    log: list[str] = []
    app = create_app(
        settings=server_settings,
        capabilities=[Recording("db", log), Recording("broker", log)],
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert {c["name"] for c in response.json()["checks"]} == {"db", "broker"}


def test_a_failing_capability_makes_the_service_unready(server_settings: Any) -> None:
    log: list[str] = []
    app = create_app(
        settings=server_settings,
        capabilities=[Recording("db", log, status=ProbeStatus.FAILED)],
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "failed"


def test_a_degraded_capability_still_takes_traffic(server_settings: Any) -> None:
    log: list[str] = []
    app = create_app(
        settings=server_settings,
        capabilities=[Recording("prompts", log, status=ProbeStatus.DEGRADED)],
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_a_probe_that_raises_is_reported_rather_than_crashing_readiness(
    server_settings: Any,
) -> None:
    log: list[str] = []
    app = create_app(settings=server_settings, capabilities=[Exploding("db", log)])

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    check = response.json()["checks"][0]
    assert check["name"] == "db"
    assert check["detail"] == "RuntimeError: probe blew up"


def test_a_service_with_no_capabilities_is_ready(server_settings: Any) -> None:
    with TestClient(create_app(settings=server_settings)) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["checks"] == []


# The server extra


def test_building_an_app_without_fastapi_names_the_extra_to_install() -> None:
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "fastapi" or name.startswith("fastapi.") or name == "base.app"
    }

    class Blocker:
        def find_spec(
            self, fullname: str, path: Any = None, target: Any = None
        ) -> None:
            if fullname == "fastapi" or fullname.startswith("fastapi."):
                raise ModuleNotFoundError(
                    f"No module named {fullname!r}", name=fullname
                )
            return None

    blocker = Blocker()
    for name in saved:
        sys.modules.pop(name, None)
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(MissingExtraError) as caught:
            importlib.import_module("base.app")
    finally:
        sys.meta_path.remove(blocker)
        for name in list(sys.modules):
            if name == "fastapi" or name.startswith("fastapi.") or name == "base.app":
                sys.modules.pop(name, None)
        sys.modules.update(saved)

    assert caught.value.extra == "server"
    assert caught.value.distribution == "fastapi"
    assert "base[server]" in str(caught.value)
    assert "create_app" in str(caught.value)
