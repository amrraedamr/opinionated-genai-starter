"""`webserver.py`: which server a service gets, and how it is configured.

One decision — uvicorn with reload locally, gunicorn everywhere else — and one
options dict that gunicorn silently discards anything it does not recognise
from. Nothing here binds a port: the uvicorn call is intercepted and the
gunicorn arbiter is replaced, so what is asserted is the configuration that
would have been served.
"""

from typing import Any

import pytest
from gunicorn.app.base import BaseApplication

from base.models.definitions.environment import Environment
from base.services.configuration.mixins.server import ServerSettingsMixin
from base.tools import webserver
from base.tools.webserver import run_gunicorn, run_uvicorn, run_webserver

APP_PATH = "base.tools.webserver:run_webserver"


@pytest.fixture
def server_settings(make_settings: Any) -> Any:
    return make_settings(
        ServerSettingsMixin,
        api_host="10.0.0.1",
        api_port=8123,
        max_gunicorn_workers=2,
        environment=Environment.PRODUCTION,
    )


@pytest.fixture
def chosen(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record which server `run_webserver` picked, without starting one."""
    picked: dict[str, Any] = {}

    def uvicorn_recorder(app_path: str, host: str = "", port: int = 0) -> None:
        picked["server"] = "uvicorn"
        picked["args"] = (app_path, host, port)

    def gunicorn_recorder(
        app_path: str, host: str = "", port: int = 0, settings: Any = None
    ) -> None:
        picked["server"] = "gunicorn"
        picked["args"] = (app_path, host, port)
        picked["settings"] = settings

    monkeypatch.setattr(webserver, "run_uvicorn", uvicorn_recorder)
    monkeypatch.setattr(webserver, "run_gunicorn", gunicorn_recorder)
    return picked


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture the configured gunicorn application instead of forking workers."""
    applications: list[Any] = []
    monkeypatch.setattr(
        BaseApplication, "run", lambda self: applications.append(self), raising=True
    )
    return applications


def test_a_local_service_is_served_by_uvicorn(
    chosen: dict[str, Any], server_settings: Any
) -> None:
    run_webserver(APP_PATH, environment=Environment.LOCAL, settings=server_settings)

    assert chosen["server"] == "uvicorn"


@pytest.mark.parametrize(
    "environment",
    [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION],
)
def test_every_deployed_environment_is_served_by_gunicorn(
    chosen: dict[str, Any], server_settings: Any, environment: Environment
) -> None:
    run_webserver(APP_PATH, environment=environment, settings=server_settings)

    assert chosen["server"] == "gunicorn"


def test_the_environment_comes_from_the_settings_when_it_is_not_given(
    chosen: dict[str, Any], server_settings: Any
) -> None:
    run_webserver(APP_PATH, settings=server_settings)

    assert chosen["server"] == "gunicorn"


def test_the_host_and_port_come_from_the_settings_when_they_are_not_given(
    chosen: dict[str, Any], server_settings: Any
) -> None:
    run_webserver(APP_PATH, settings=server_settings)

    assert chosen["args"] == (APP_PATH, "10.0.0.1", 8123)


def test_an_explicit_host_and_port_override_the_settings(
    chosen: dict[str, Any], server_settings: Any
) -> None:
    run_webserver(APP_PATH, host="127.0.0.1", port=9999, settings=server_settings)

    assert chosen["args"] == (APP_PATH, "127.0.0.1", 9999)


def test_a_port_arriving_as_a_string_is_read_as_a_number(
    chosen: dict[str, Any], server_settings: Any
) -> None:
    run_webserver(APP_PATH, port="9000", settings=server_settings)  # type: ignore[arg-type]

    assert chosen["args"][2] == 9000


def test_port_zero_is_honoured_rather_than_read_as_unset(
    chosen: dict[str, Any], server_settings: Any
) -> None:
    """`0` asks the kernel for a free port, so it cannot fall back to settings."""
    run_webserver(APP_PATH, port=0, settings=server_settings)

    assert chosen["args"][2] == 0


def test_the_registered_settings_are_used_when_none_are_passed(
    chosen: dict[str, Any], make_settings: Any
) -> None:
    make_settings(ServerSettingsMixin, api_host="0.0.0.0", api_port=8080)

    run_webserver(APP_PATH)

    assert chosen["server"] == "uvicorn"
    assert chosen["args"] == (APP_PATH, "0.0.0.0", 8080)


def test_settings_without_the_server_mixin_are_refused(make_settings: Any) -> None:
    from base.tools.errors import MissingSettingsError

    make_settings()
    with pytest.raises(MissingSettingsError, match="SupportsServerConfig"):
        run_webserver(APP_PATH)


def test_a_port_that_is_not_a_number_is_refused_before_anything_is_served(
    chosen: dict[str, Any], server_settings: Any
) -> None:
    with pytest.raises(ValueError):
        run_webserver(APP_PATH, port="http", settings=server_settings)  # type: ignore[arg-type]
    assert chosen == {}


def test_the_settings_are_handed_on_to_gunicorn(
    chosen: dict[str, Any], server_settings: Any
) -> None:
    run_webserver(APP_PATH, settings=server_settings)

    assert chosen["settings"] is server_settings


def test_uvicorn_is_started_with_reload_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reload is the whole reason local differs; it is not a setting."""
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(
        webserver.uvicorn,
        "run",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    run_uvicorn(APP_PATH, "127.0.0.1", 8001)

    assert calls == [(APP_PATH, {"host": "127.0.0.1", "port": 8001, "reload": True})]


def test_uvicorn_reads_a_port_that_arrived_as_a_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        webserver.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs)
    )

    run_uvicorn(APP_PATH, port="8002")  # type: ignore[arg-type]

    assert calls[0]["port"] == 8002


def test_uvicorn_binds_every_interface_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        webserver.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs)
    )

    run_uvicorn(APP_PATH)

    assert (calls[0]["host"], calls[0]["port"]) == ("0.0.0.0", 8000)


def test_gunicorn_binds_the_host_and_port_it_was_given(
    served: list[Any], server_settings: Any
) -> None:
    run_gunicorn(APP_PATH, "1.2.3.4", 9000, settings=server_settings)

    assert served[0].cfg.bind == ["1.2.3.4:9000"]


def test_gunicorn_serves_the_application_the_path_names(
    served: list[Any], server_settings: Any
) -> None:
    run_gunicorn(APP_PATH, settings=server_settings)

    assert served[0].load() is run_webserver


def test_an_app_path_that_names_nothing_fails_before_a_worker_is_forked(
    served: list[Any], server_settings: Any
) -> None:
    with pytest.raises(ValueError, match="Expected 'package.module:attribute'"):
        run_gunicorn("base.tools.webserver", settings=server_settings)
    assert served == []


def test_the_worker_count_never_exceeds_the_configured_ceiling(
    served: list[Any], make_settings: Any
) -> None:
    settings = make_settings(ServerSettingsMixin, max_gunicorn_workers=1)

    run_gunicorn(APP_PATH, settings=settings)

    assert served[0].cfg.workers == 1


def test_the_worker_count_follows_the_machine_when_the_ceiling_is_high(
    served: list[Any], make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(webserver.multiprocessing, "cpu_count", lambda: 3)
    settings = make_settings(ServerSettingsMixin, max_gunicorn_workers=99)

    run_gunicorn(APP_PATH, settings=settings)

    assert served[0].cfg.workers == 4


def test_gunicorn_runs_the_asgi_worker(served: list[Any], server_settings: Any) -> None:
    run_gunicorn(APP_PATH, settings=server_settings)

    assert served[0].cfg.settings["worker_class"].get() == (
        "uvicorn.workers.UvicornWorker"
    )


def test_the_application_is_loaded_before_the_workers_fork(
    served: list[Any], server_settings: Any
) -> None:
    """Gunicorn names the setting `preload_app` and drops any key it does not
    recognise, so `preload` was accepted and then silently discarded."""
    run_gunicorn(APP_PATH, settings=server_settings)

    assert served[0].cfg.preload_app is True


def test_the_request_timeouts_reach_gunicorn(
    served: list[Any], server_settings: Any
) -> None:
    run_gunicorn(APP_PATH, settings=server_settings)

    assert (served[0].cfg.timeout, served[0].cfg.graceful_timeout) == (500, 300)


def test_gunicorn_falls_back_to_the_registered_settings(
    served: list[Any], make_settings: Any
) -> None:
    make_settings(ServerSettingsMixin, max_gunicorn_workers=1)

    run_gunicorn(APP_PATH)

    assert served[0].cfg.workers == 1


def test_an_option_gunicorn_does_not_recognise_is_dropped_rather_than_raising(
    served: list[Any], server_settings: Any
) -> None:
    run_gunicorn(APP_PATH, settings=server_settings)
    application = served[0]
    application.options["not_a_gunicorn_setting"] = "x"

    application.load_config()

    assert not hasattr(application.cfg, "not_a_gunicorn_setting")


def test_an_option_left_unset_does_not_overwrite_a_gunicorn_default(
    served: list[Any], server_settings: Any
) -> None:
    run_gunicorn(APP_PATH, settings=server_settings)
    application = served[0]
    application.options["timeout"] = None

    application.load_config()

    assert application.cfg.timeout == 500


def test_configuring_before_gunicorn_has_built_its_config_is_a_no_op(
    served: list[Any], server_settings: Any
) -> None:
    run_gunicorn(APP_PATH, settings=server_settings)
    application = served[0]
    application.cfg = None

    application.load_config()

    assert application.cfg is None
