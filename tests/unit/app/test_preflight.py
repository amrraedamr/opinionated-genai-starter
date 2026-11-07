"""Preflight answers one question: can this image run what it is configured to run?

Capability is fixed at build time by extras; the implementation is chosen later
by environment variables, and nothing else reconciles the two. A container that
starts anyway and quietly does the wrong thing is the failure this prevents, so
an unrecognised value is fatal rather than a fallback.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from base import preflight
from base.models.definitions.broker import BrokerBackend
from base.models.definitions.database import SQLBackend


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SQL_BACKEND", "BROKER", "ENVIRONMENT"):
        monkeypatch.delenv(name, raising=False)


def block(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make `find_spec` report the named modules as absent."""
    real = importlib.util.find_spec

    def fake(name: str, package: str | None = None) -> Any:
        return None if name in names else real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


# SQL


def test_a_build_without_sqlalchemy_does_not_check_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block(monkeypatch, "sqlalchemy")
    monkeypatch.setenv("SQL_BACKEND", "nonsense")

    assert preflight.check_sql() == []


def test_the_configured_backend_passes_when_its_drivers_are_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SQL_BACKEND", "sqlite")

    assert preflight.check_sql() == []
    assert "SQL OK: backend 'sqlite'" in capsys.readouterr().out


def test_an_unset_backend_is_checked_as_the_settings_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert preflight.check_sql() == []
    assert "backend 'sqlite'" in capsys.readouterr().out


def test_an_unknown_backend_is_refused_and_the_valid_values_are_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SQL_BACKEND", "oracle")

    failures = preflight.check_sql()

    assert len(failures) == 1
    assert "SQL_BACKEND='oracle' is not a known value" in failures[0]
    for member in SQLBackend:
        assert member.value in failures[0]


def test_a_backend_whose_driver_is_missing_names_the_extra_to_build_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block(monkeypatch, "asyncpg", "psycopg2")
    monkeypatch.setenv("SQL_BACKEND", "postgresql")

    failures = preflight.check_sql()

    assert len(failures) == 1
    assert "SQL_BACKEND='postgresql' needs psycopg2, asyncpg" in failures[0]
    assert "base[postgres]" in failures[0]
    assert "asyncpg / psycopg2-binary" in failures[0]


def test_the_failure_says_which_backends_the_image_does_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block(monkeypatch, "asyncpg", "psycopg2", "aiomysql", "pymysql")
    monkeypatch.setenv("SQL_BACKEND", "postgresql")

    failures = preflight.check_sql()

    assert "This build supports: sqlite" in failures[0]


# Broker


def test_the_default_transport_passes_in_local_development(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert preflight.check_broker() == []
    assert "Broker OK: transport 'memory'" in capsys.readouterr().out


def test_an_unknown_transport_is_refused_rather_than_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROKER", "kafka")

    failures = preflight.check_broker()

    assert len(failures) == 1
    assert "BROKER='kafka' is not a known value" in failures[0]
    for member in BrokerBackend:
        assert member.value in failures[0]


def test_a_transport_whose_client_is_missing_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block(monkeypatch, "redis")
    monkeypatch.setenv("BROKER", "redis")

    failures = preflight.check_broker()

    assert len(failures) == 1
    assert "BROKER='redis' needs redis" in failures[0]
    assert "base[redis]" in failures[0]
    assert "This build supports: memory" in failures[0]


def test_an_in_process_transport_outside_local_is_warned_about_loudly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("BROKER", "memory")
    monkeypatch.setenv("ENVIRONMENT", "production")

    failures = preflight.check_broker()
    captured = capsys.readouterr()

    assert failures == []
    assert "WARNING: BROKER='memory' in environment 'production'" in captured.err
    assert "Set BROKER=redis" in captured.err


def test_a_distributed_transport_outside_local_passes_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("BROKER", "redis")
    monkeypatch.setenv("ENVIRONMENT", "production")

    failures = preflight.check_broker()
    captured = capsys.readouterr()

    assert failures == []
    assert "Broker OK: transport 'redis'" in captured.out
    assert captured.err == ""


# main


def test_a_build_that_can_run_its_configuration_exits_zero() -> None:
    assert preflight.main() == 0


def test_any_failure_exits_non_zero_and_is_printed_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SQL_BACKEND", "oracle")

    code = preflight.main()

    assert code == 1
    assert "Preflight failed: SQL_BACKEND='oracle'" in capsys.readouterr().err


def test_every_failing_check_is_reported_not_just_the_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SQL_BACKEND", "oracle")
    monkeypatch.setenv("BROKER", "kafka")

    assert preflight.main() == 1

    errors = capsys.readouterr().err
    assert "SQL_BACKEND='oracle'" in errors
    assert "BROKER='kafka'" in errors


def test_the_settings_default_is_read_from_the_mixin_not_hard_coded() -> None:
    from base.services.configuration.mixins import BrokerSettingsMixin, SQLSettingsMixin

    assert preflight._default(SQLSettingsMixin, "sql_backend") == "sqlite"
    assert preflight._default(BrokerSettingsMixin, "broker") == "memory"
