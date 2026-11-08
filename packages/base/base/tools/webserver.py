import importlib
import logging
import multiprocessing
from typing import Any

from base.models.definitions.environment import Environment
from base.services.configuration import settings_for
from base.tools.errors import MissingExtraError

try:
    import uvicorn
    from gunicorn.app.base import BaseApplication
    from starlette.types import ASGIApp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "server", "uvicorn / gunicorn", feature="run_webserver"
    ) from exc

from base.services.configuration.protocols import SupportsServerConfig


def _import_app(app_path: str) -> Any:
    """Resolve an ASGI application from a `"package.module:attribute"` string."""
    module_name, separator, attribute = app_path.partition(":")
    if not separator or not attribute:
        raise ValueError(
            f"Invalid app path {app_path!r}. Expected 'package.module:attribute'."
        )
    return getattr(importlib.import_module(module_name), attribute)


def run_webserver(
    app_path: str,
    host: str | None = None,
    port: int | None = None,
    environment: Environment | None = None,
    settings: SupportsServerConfig | None = None,
) -> None:
    """Serve `app_path` with uvicorn locally and gunicorn everywhere else."""
    settings = settings or settings_for(SupportsServerConfig)
    host = host or settings.api_host
    port = int(port if port is not None else settings.api_port)
    environment = environment if environment is not None else settings.environment

    logging.info(f"Starting webserver (environment={environment})")
    if environment == Environment.LOCAL:
        run_uvicorn(app_path, host, port)
    else:
        run_gunicorn(app_path, host, port, settings=settings)


def run_uvicorn(app_path: str, host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(
        app_path,
        host=host,
        port=int(port),
        reload=True,
    )


def run_gunicorn(
    app_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    settings: SupportsServerConfig | None = None,
) -> None:
    settings = settings or settings_for(SupportsServerConfig)

    # Runs pre-fork under `preload: True`. Create engines, pools and clients
    # in a capability's `startup()`, which runs per worker post-fork.
    app = _import_app(app_path)

    class AIPlatform(BaseApplication):
        def __init__(self, app: ASGIApp, options: dict[str, Any] | None = None) -> None:
            self.options: dict[str, Any] = options or {}
            self.application = app
            super().__init__()

        def load_config(self) -> None:
            cfg = self.cfg
            if cfg is None:
                return

            config = {
                key: value
                for key, value in self.options.items()
                if key in cfg.settings and value is not None
            }
            for key, value in config.items():
                cfg.set(key.lower(), value)

        def load(self) -> ASGIApp:
            return self.application

    options = {
        "bind": f"{host}:{port}",
        "workers": min(multiprocessing.cpu_count() + 1, settings.max_gunicorn_workers),
        "worker_class": "uvicorn.workers.UvicornWorker",
        "preload": True,
        "timeout": 500,
        "graceful_timeout": 300,
    }

    AIPlatform(app, options).run()
