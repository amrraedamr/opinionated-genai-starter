"""`ChatApiSettings` is this service's whole statement of what it needs.

A subsystem reads settings through a protocol, so a mixin the service forgot
to compose is a composition mistake rather than a missing attribute later.
The protocols are therefore checked directly, against the subsystems this
service actually reaches: SQL, the broker, token verification, the LLM and
the prompt store.
"""

from __future__ import annotations

import pytest

from base.services.common.meta import purge_all
from base.services.configuration import configure, settings_for
from base.services.configuration.protocols import (
    SupportsBrokerConfig,
    SupportsLLMConfig,
    SupportsPostgresConfig,
    SupportsPromptConfig,
    SupportsRedisConfig,
    SupportsServerConfig,
    SupportsSQLConfig,
    SupportsTokenVerificationConfig,
)
from chat_api.settings import ChatApiSettings


@pytest.fixture
def settings() -> ChatApiSettings:
    """Settings built from nothing but their defaults.

    `CoreSettings` is a singleton, so one built in an earlier test would
    answer here; `_env_file=None` keeps a developer's own `.env` out.
    """
    purge_all()
    return ChatApiSettings(_env_file=None)


# Composition


@pytest.mark.parametrize(
    "contract",
    [
        SupportsServerConfig,
        SupportsSQLConfig,
        SupportsPostgresConfig,
        SupportsRedisConfig,
        SupportsBrokerConfig,
        SupportsLLMConfig,
        SupportsTokenVerificationConfig,
        SupportsPromptConfig,
    ],
)
def test_the_settings_satisfy_every_contract_this_service_reads(
    settings: ChatApiSettings, contract: type
) -> None:
    configure(settings)
    assert settings_for(contract) is settings


def test_the_service_does_not_claim_to_issue_tokens(
    settings: ChatApiSettings,
) -> None:
    """Only the identity service signs; this one verifies."""
    from base.services.configuration.protocols import SupportsTokenIssuingConfig

    assert not isinstance(settings, SupportsTokenIssuingConfig)


# Defaults


def test_the_service_names_itself(settings: ChatApiSettings) -> None:
    assert settings.service_name == "chat_api"


def test_the_api_prefix_is_the_services_own(settings: ChatApiSettings) -> None:
    assert settings.api_prefix == "/api/7r"


def test_no_origin_is_allowed_until_one_is_configured(
    settings: ChatApiSettings,
) -> None:
    assert settings.cors_origins == ""
    assert settings.cors_origins_list == []


# CORS origins


def test_origins_are_split_on_commas_and_stripped() -> None:
    purge_all()
    settings = ChatApiSettings(
        _env_file=None,
        cors_origins=" https://a.example.com , https://b.example.com ",
    )
    assert settings.cors_origins_list == [
        "https://a.example.com",
        "https://b.example.com",
    ]


def test_a_single_origin_still_reads_as_a_list() -> None:
    purge_all()
    settings = ChatApiSettings(_env_file=None, cors_origins="https://a.example.com")
    assert settings.cors_origins_list == ["https://a.example.com"]


def test_blank_entries_are_dropped_rather_than_allowed() -> None:
    """An empty origin would otherwise read as one that matches nothing."""
    purge_all()
    settings = ChatApiSettings(_env_file=None, cors_origins="https://a.example.com,,")
    assert settings.cors_origins_list == ["https://a.example.com"]


# The environment


def test_a_setting_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_PREFIX", "/api/v9")
    purge_all()
    assert ChatApiSettings(_env_file=None).api_prefix == "/api/v9"


def test_an_environment_variable_the_service_does_not_declare_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment sets variables for the whole fleet, not for one service."""
    monkeypatch.setenv("SOMETHING_ANOTHER_SERVICE_NEEDS", "x")
    purge_all()
    assert ChatApiSettings(_env_file=None).service_name == "chat_api"


def test_the_settings_are_one_instance_per_process(
    settings: ChatApiSettings,
) -> None:
    assert ChatApiSettings(_env_file=None) is settings


def test_the_module_level_settings_are_the_ones_the_routers_read() -> None:
    from chat_api import settings as settings_module

    assert isinstance(settings_module.settings, ChatApiSettings)
    assert settings_module.settings.service_name
