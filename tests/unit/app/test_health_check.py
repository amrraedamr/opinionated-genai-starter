"""The probe a container runs to decide whether this process is still alive.

It runs as `python -m base.health_check` with no package imported, so what
matters is the exit code and the url it builds: a wrong host means a restart
loop on a service that was perfectly healthy.
"""

from __future__ import annotations

import runpy
import urllib.error
import urllib.request
from typing import Any

import pytest

from base import health_check


class Answer:
    """What `urlopen` hands back, as a context manager."""

    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "Answer":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class Probe:
    """Stands in for `urlopen`, keeping the url and timeout it was asked for."""

    def __init__(self, status: int = 200, error: Exception | None = None) -> None:
        self.status = status
        self.error = error
        self.url = ""
        self.timeout: float | None = None

    def __call__(self, url: str, timeout: float | None = None) -> Answer:
        self.url = url
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        return Answer(self.status)


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Installs a stand-in for `urlopen` and hands it back."""

    def _install(status: int = 200, error: Exception | None = None) -> Probe:
        stub = Probe(status, error)
        monkeypatch.setattr(urllib.request, "urlopen", stub)
        return stub

    return _install


@pytest.fixture(autouse=True)
def _no_inherited_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)


# The verdict


def test_a_service_answering_two_hundred_is_healthy(probe: Any) -> None:
    probe(status=200)

    assert health_check.main() == 0


@pytest.mark.parametrize("status", [201, 204, 301, 404, 500, 503])
def test_anything_other_than_two_hundred_is_unhealthy(probe: Any, status: int) -> None:
    probe(status=status)

    assert health_check.main() == 1


def test_a_refused_connection_is_unhealthy(probe: Any) -> None:
    probe(error=urllib.error.URLError(ConnectionRefusedError("refused")))

    assert health_check.main() == 1


def test_a_probe_that_timed_out_is_unhealthy(probe: Any) -> None:
    probe(error=TimeoutError("took too long"))

    assert health_check.main() == 1


def test_an_error_nobody_expected_is_unhealthy_rather_than_a_traceback(
    probe: Any,
) -> None:
    """A probe that raises is read by the orchestrator as a crash, not a verdict."""
    probe(error=ValueError("something else entirely"))

    assert health_check.main() == 1


# The url


def test_the_probe_asks_the_liveness_endpoint(probe: Any) -> None:
    stub = probe()

    health_check.main()

    assert stub.url == "http://127.0.0.1:8000/livez"


def test_the_host_and_port_come_from_the_environment(
    probe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_HOST", "api.internal")
    monkeypatch.setenv("API_PORT", "9443")
    stub = probe()

    health_check.main()

    assert stub.url == "http://api.internal:9443/livez"


def test_a_wildcard_bind_address_is_probed_on_the_loopback(
    probe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing can connect *to* 0.0.0.0, so the container asks itself instead."""
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    stub = probe()

    health_check.main()

    assert stub.url == "http://127.0.0.1:8000/livez"


def test_a_host_that_only_looks_like_a_wildcard_is_left_alone(
    probe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_HOST", "0.0.0.0.example.com")
    stub = probe()

    health_check.main()

    assert stub.url == "http://0.0.0.0.example.com:8000/livez"


def test_the_probe_gives_up_rather_than_hanging(probe: Any) -> None:
    stub = probe()

    health_check.main()

    assert stub.timeout == 4


# As a script


def test_running_the_module_exits_with_what_the_probe_decided(probe: Any) -> None:
    probe(status=200)

    with pytest.raises(SystemExit) as exit_code:
        runpy.run_path(health_check.__file__, run_name="__main__")

    assert exit_code.value.code == 0


def test_running_the_module_against_a_dead_service_exits_non_zero(
    probe: Any,
) -> None:
    probe(status=503)

    with pytest.raises(SystemExit) as exit_code:
        runpy.run_path(health_check.__file__, run_name="__main__")

    assert exit_code.value.code == 1
