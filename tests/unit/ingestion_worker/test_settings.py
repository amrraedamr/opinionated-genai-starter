"""`IngestionWorkerSettings` is this service's whole statement of what it needs.

A subsystem reads settings through a protocol, so a mixin the service forgot
to compose is a composition mistake rather than a missing attribute later.
The protocols are therefore checked directly, against the two subsystems this
service actually reaches: its own SQL inbox and the broker it consumes from.
"""

from __future__ import annotations

import pytest

from base.models.definitions.broker import BrokerBackend
from base.models.definitions.environment import Environment
from base.services.common.meta import purge_all
from base.services.configuration import configure, settings_for
from base.services.configuration.protocols import (
    SupportsBrokerConfig,
    SupportsPostgresConfig,
    SupportsRedisBrokerConfig,
    SupportsRedisConfig,
    SupportsSQLConfig,
)
from ingestion_worker.settings import IngestionWorkerSettings


@pytest.fixture
def settings() -> IngestionWorkerSettings:
    """Settings built from nothing but their defaults."""
    purge_all()
    return IngestionWorkerSettings(_env_file=None)


# Composition


@pytest.mark.parametrize(
    "contract",
    [
        SupportsSQLConfig,
        SupportsPostgresConfig,
        SupportsRedisConfig,
        SupportsBrokerConfig,
        SupportsRedisBrokerConfig,
    ],
)
def test_the_settings_satisfy_every_contract_this_service_reads(
    settings: IngestionWorkerSettings, contract: type
) -> None:
    configure(settings)
    assert settings_for(contract) is settings


def test_the_service_does_not_claim_an_http_surface(
    settings: IngestionWorkerSettings,
) -> None:
    """A worker consumes events and serves nothing, so `create_app` must refuse it."""
    from base.services.configuration.protocols import SupportsServerConfig

    assert not isinstance(settings, SupportsServerConfig)


def test_the_service_neither_issues_nor_verifies_tokens(
    settings: IngestionWorkerSettings,
) -> None:
    from base.services.configuration.protocols import (
        SupportsTokenIssuingConfig,
        SupportsTokenVerificationConfig,
    )

    assert not isinstance(settings, SupportsTokenIssuingConfig)
    assert not isinstance(settings, SupportsTokenVerificationConfig)


# Defaults


def test_the_service_names_itself(settings: IngestionWorkerSettings) -> None:
    assert settings.service_name == "ingestion_worker"


def test_the_consumer_group_is_the_services_own(
    settings: IngestionWorkerSettings,
) -> None:
    """Every replica shares one group, so one of them handles each event."""
    assert settings.consumer_group == "ingestion_worker"


def test_the_inbox_lives_in_the_services_own_file(
    settings: IngestionWorkerSettings,
) -> None:
    assert settings.sqlite_db_path == "database/ingestion_worker.db"


def test_a_claim_is_held_for_five_minutes_before_another_replica_may_take_it(
    settings: IngestionWorkerSettings,
) -> None:
    assert settings.inbox_claim_timeout_seconds == 300


def test_the_transport_defaults_to_the_in_process_one(
    settings: IngestionWorkerSettings,
) -> None:
    """Which is correct locally and never correct across containers."""
    assert settings.broker == BrokerBackend.MEMORY
    assert settings.broker.is_distributed is False


def test_the_delivery_policy_comes_from_the_broker_mixin(
    settings: IngestionWorkerSettings,
) -> None:
    assert settings.broker_max_attempts == 5
    assert settings.broker_claim_idle_ms == 30_000
    assert settings.broker_retry_backoff_seconds == 0.25


def test_the_environment_defaults_to_local(settings: IngestionWorkerSettings) -> None:
    assert settings.environment == Environment.LOCAL


# The environment


def test_a_setting_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONSUMER_GROUP", "ingestion_worker_canary")
    purge_all()
    settings = IngestionWorkerSettings(_env_file=None)
    assert settings.consumer_group == "ingestion_worker_canary"


def test_the_claim_timeout_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INBOX_CLAIM_TIMEOUT_SECONDS", "45")
    purge_all()
    assert IngestionWorkerSettings(_env_file=None).inbox_claim_timeout_seconds == 45


def test_a_claim_timeout_that_is_not_a_number_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INBOX_CLAIM_TIMEOUT_SECONDS", "soon")
    purge_all()
    with pytest.raises(ValueError):
        IngestionWorkerSettings(_env_file=None)


def test_a_transport_this_library_does_not_know_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROKER", "kafka")
    purge_all()
    with pytest.raises(ValueError):
        IngestionWorkerSettings(_env_file=None)


def test_an_environment_variable_the_service_does_not_declare_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment sets variables for the whole fleet, not for one service."""
    monkeypatch.setenv("SOMETHING_ANOTHER_SERVICE_NEEDS", "x")
    purge_all()
    assert IngestionWorkerSettings(_env_file=None).service_name == "ingestion_worker"


def test_the_settings_are_one_instance_per_process(
    settings: IngestionWorkerSettings,
) -> None:
    assert IngestionWorkerSettings(_env_file=None) is settings


def test_the_module_level_settings_are_the_ones_the_worker_reads() -> None:
    from ingestion_worker import settings as settings_module

    assert isinstance(settings_module.settings, IngestionWorkerSettings)
    assert settings_module.settings.service_name
