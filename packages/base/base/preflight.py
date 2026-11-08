"""Startup preflight: can this build run what it is configured to run?

    python -m base.preflight

A subsystem's capability is fixed when the image is built, by the extras its
service declares. Which implementation it uses is chosen later, by environment
variables. Nothing reconciles the two, so configuration can name a backend
whose dependency was never installed — or, worse, name nothing recognisable
and fall through to a default that starts cleanly and does the wrong thing.

`docker/run.sh` runs this before migrations and before `serve`, so a container
fails at boot with a reason an operator can act on, and fails identically
whether a service builds its clients eagerly at import or lazily on first use.

Checks:
  SQL     — whenever the build carries the `sql` extra, which is what makes a
            service speak SQL at all. Not `alembic.ini`: that marks migrations,
            and a service can own tables whose schema is managed elsewhere.
  Broker  — always. An unrecognised value is fatal, because the alternative is
            falling back to an in-process transport that silently drops every
            cross-service event while reporting healthy.

Ask an image what it supports without starting it:

    docker run --rm --entrypoint python <image> -m base.preflight
"""

import importlib.util
import os
import sys
from enum import StrEnum

from base.models.definitions.broker import BrokerBackend, installed_brokers
from base.models.definitions.database import SQLBackend, installed_backends
from base.models.definitions.environment import Environment
from base.services.configuration.core import CoreSettings
from base.services.configuration.mixins import (
    BrokerSettingsMixin,
    SQLSettingsMixin,
)
from base.tools.errors import UnsupportedBackendError


def _warn(message: str) -> None:
    """Write to stderr after stdout, so container logs stay in order."""
    sys.stdout.flush()
    print(message, file=sys.stderr)
    sys.stderr.flush()


def _default(model: type, field: str) -> str:
    """Settings default, so an unset variable is checked as the service resolves it."""
    return str(model.model_fields[field].default)


def _unknown_value(variable: str, value: str, known: type[StrEnum]) -> str:
    options = ", ".join(member.value for member in known)
    return f"{variable}={value!r} is not a known value. Valid values: {options}."


def check_sql() -> list[str]:
    """Validate `SQL_BACKEND` against the drivers this build installs.

    Skipped only when the build has no `sql` extra, in which case the service
    does not speak SQL and `SQL_BACKEND` is inert. `alembic.ini` is deliberately
    not the signal: it marks migrations, so gating on it would skip a SQL
    service whose schema is managed outside the repository.
    """
    if importlib.util.find_spec("sqlalchemy") is None:
        return []

    configured = os.environ.get("SQL_BACKEND") or _default(
        SQLSettingsMixin, "sql_backend"
    )
    try:
        backend = SQLBackend(configured)
    except ValueError:
        return [_unknown_value("SQL_BACKEND", configured, SQLBackend)]

    if backend.is_installed:
        supported = ", ".join(member.value for member in installed_backends())
        print(f"SQL OK: backend {backend.value!r}; this build supports {supported}")
        return []

    return [
        str(
            UnsupportedBackendError(
                backend=backend.value,
                extra=backend.extra,
                distribution=backend.distribution,
                missing=backend.missing_imports,
                available=[member.value for member in installed_backends()],
            )
        )
    ]


def check_broker() -> list[str]:
    """Validate `BROKER` against the transports this build installs.

    An unknown value is an error rather than a fallback. `build_broker()`
    treats anything that is not an exact match as in-process, which starts
    cleanly, passes its readiness probe, and delivers nothing between
    containers.
    """
    configured = os.environ.get("BROKER") or _default(BrokerSettingsMixin, "broker")
    try:
        backend = BrokerBackend(configured)
    except ValueError:
        return [_unknown_value("BROKER", configured, BrokerBackend)]

    if not backend.is_installed:
        return [
            f"BROKER={backend.value!r} needs {', '.join(backend.missing_imports)}, "
            f"which this build does not install. This build supports: "
            f"{', '.join(member.value for member in installed_brokers())}. Either set "
            f"BROKER to one of those, or deploy an image built with "
            f'"base[{backend.extra}]" (installs {backend.distribution}).'
        ]

    # Read raw: preflight runs before settings are built, so an unrecognised
    # name must warn rather than raise.
    raw = os.environ.get("ENVIRONMENT") or _default(CoreSettings, "environment")
    environment = Environment(raw) if raw in set(Environment) else None
    if not backend.is_distributed and environment != Environment.LOCAL:
        _warn(
            f"WARNING: BROKER={backend.value!r} in environment {raw!r}. "
            f"This transport is in-process, so each replica owns a private queue "
            f"and events reach no other service. Set BROKER=redis unless this "
            f"deployment is deliberately a single process."
        )
    else:
        print(f"Broker OK: transport {backend.value!r}")
    return []


def main() -> int:
    failures: list[str] = []

    failures += check_sql()
    failures += check_broker()

    for failure in failures:
        _warn(f"Preflight failed: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
