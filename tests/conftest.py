"""Fixtures every test in this repository builds on.

Three kinds of test live here, and the difference is what they need rather
than what they cover:

* unit         nothing outside the process; always runs
* integration  the containers in `deploy/docker-compose.infra.yml`
* live         real provider credentials in `.env`

The latter two skip themselves when what they need is absent, so a clone with
no Docker and no keys still runs the whole unit suite and reports honestly on
the rest. Nothing is silently passed.
"""

import os
import pathlib
import socket
import urllib.parse
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from base.services.configuration import configure, reset
from base.services.configuration.core import CoreSettings


def _load_dotenv() -> dict[str, str]:
    """`.env` as a mapping, without disturbing the real environment."""
    path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


DOTENV: dict[str, str] = _load_dotenv()


def credential(name: str) -> str:
    """A credential from the environment, falling back to `.env`."""
    return os.environ.get(name) or DOTENV.get(name, "")


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """Whether something is listening, without importing a client library."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _host_resolves(url: str) -> bool:
    """A corporate gateway often resolves only on a VPN."""
    host = urllib.parse.urlsplit(url).hostname
    if not host:
        return False
    try:
        socket.gethostbyname(host)
        return True
    except OSError:
        return False


CONTAINERS: dict[str, tuple[str, int]] = {
    "postgres": ("localhost", 5432),
    "redis": ("localhost", 6379),
    "mongo": ("localhost", 27017),
    "qdrant": ("localhost", 6333),
    "azurite": ("localhost", 10100),
    "cosmos": ("localhost", 8081),
    "otel-collector": ("localhost", 4317),
    "jaeger": ("localhost", 16686),
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: needs local containers")
    config.addinivalue_line("markers", "live: needs provider credentials")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip what cannot run here, and say which piece is missing."""
    missing = [n for n, (h, p) in CONTAINERS.items() if not _port_open(h, p)]
    containers_down = pytest.mark.skip(
        reason=f"containers not reachable: {', '.join(missing)}. "
        "Start them with: docker compose -f deploy/docker-compose.infra.yml up -d"
    )
    endpoint = credential("OPENAI_LLM_ENDPOINT")
    key = credential("OPENAI_LLM_API_KEY")
    if not (endpoint and key):
        live_reason = "no provider credentials; copy .env.example to .env"
    elif not _host_resolves(endpoint):
        live_reason = f"{urllib.parse.urlsplit(endpoint).hostname} does not resolve"
    else:
        live_reason = ""
    no_credentials = pytest.mark.skip(reason=live_reason)

    for item in items:
        if missing and "integration" in item.keywords:
            item.add_marker(containers_down)
        if live_reason and "live" in item.keywords:
            item.add_marker(no_credentials)


@pytest.fixture(autouse=True)
def _isolated_settings() -> Iterator[None]:
    """Drop registered settings and cached services around every test.

    `CoreSettings` is a singleton and `configure` registers one process-wide,
    so without this a service configured in one test would answer in the next.
    Services are singletons too, and an async one caches a client bound to the
    event loop it was built on — which the next test has already closed.
    """
    from base.services.common.meta import purge_all

    purge_all()
    reset()
    yield
    reset()
    purge_all()


def settings_class(*mixins: type) -> type[CoreSettings]:
    """A settings class composed of exactly the mixins a test names.

    Composition is the thing being tested in half these cases, so the class is
    built per test rather than shared. `env_file=None` keeps a developer's own
    `.env` from reaching a unit test.
    """

    class Settings(CoreSettings, *mixins):  # type: ignore[misc]
        model_config = SettingsConfigDict(env_file=None, extra="ignore")

    return Settings


@pytest.fixture
def make_settings() -> Any:
    """Build and register settings for one test."""

    def _make(*mixins: type, **values: Any) -> CoreSettings:
        values.setdefault("service_name", "test-service")
        return configure(settings_class(*mixins)(**values))

    return _make


@pytest.fixture
def infra_settings(make_settings: Any) -> CoreSettings:
    """Settings pointing at the local containers."""
    from base.services.configuration.mixins import (
        BrokerSettingsMixin,
        MongoSettingsMixin,
        PostgresSettingsMixin,
        QdrantSettingsMixin,
        RateLimitSettingsMixin,
        RedisSettingsMixin,
        SQLSettingsMixin,
    )

    return make_settings(
        SQLSettingsMixin,
        PostgresSettingsMixin,
        RedisSettingsMixin,
        MongoSettingsMixin,
        QdrantSettingsMixin,
        BrokerSettingsMixin,
        RateLimitSettingsMixin,
        sql_backend="postgresql",
        pg_host="localhost",
        pg_port="5432",
        pg_database="postgres",
        pg_username="local",
        pg_password="local",
        db_ssl_enabled=False,
        redis_url="redis://localhost:6379/1",
        mongodb_host="localhost:27017",
        mongodb_database=f"test_{os.getpid()}",
        qdrant_url="http://localhost:6333",
        broker="redis",
        rate_limit_auth_per_minute=5,
        rate_limit_llm_tokens_per_minute=100,
    )


@pytest.fixture
def unique() -> str:
    """A short token for naming a table, collection or key uniquely."""
    import uuid

    return uuid.uuid4().hex[:10]
