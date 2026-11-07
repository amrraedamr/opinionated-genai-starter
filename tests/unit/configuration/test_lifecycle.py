"""The settings registry: configure, get, is_configured, reset.

Pins that one process holds exactly one settings object, that reading before
one is registered says what to do about it, and that `CoreSettings`'s
singleton metaclass makes a second construction of the same class hand back
the first instance — the trap every test in this area has to work around.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from base.services.configuration import (
    configure,
    get_settings,
    is_configured,
    reset,
)
from base.services.configuration.core import CoreSettings
from base.services.configuration.mixins import RedisSettingsMixin, SQLSettingsMixin
from tests.conftest import settings_class


def test_nothing_is_configured_at_the_start_of_a_test() -> None:
    assert is_configured() is False


def test_get_settings_before_configure_names_the_call_that_fixes_it() -> None:
    with pytest.raises(RuntimeError) as caught:
        get_settings()
    message = str(caught.value)
    assert "Settings not configured" in message
    assert "base.services.configuration.configure" in message


def test_configure_hands_back_the_instance_it_registered() -> None:
    instance = settings_class(RedisSettingsMixin)(service_name="svc")
    assert configure(instance) is instance
    assert get_settings() is instance


def test_is_configured_follows_configure_and_reset() -> None:
    configure(settings_class(RedisSettingsMixin)(service_name="svc"))
    assert is_configured() is True
    reset()
    assert is_configured() is False


def test_reset_leaves_the_next_read_unconfigured() -> None:
    configure(settings_class(RedisSettingsMixin)(service_name="svc"))
    reset()
    with pytest.raises(RuntimeError, match="Settings not configured"):
        get_settings()


def test_configure_replaces_whatever_was_registered_before() -> None:
    first = settings_class(RedisSettingsMixin)(service_name="first")
    second = settings_class(SQLSettingsMixin)(service_name="second")
    configure(first)
    configure(second)
    assert get_settings() is second


def test_constructing_one_settings_class_twice_returns_one_object() -> None:
    """`SingletonSettings` means the second call's values are discarded."""
    cls = settings_class(RedisSettingsMixin)
    first = cls(service_name="first", redis_url="redis://one:6379/0")
    second = cls(service_name="second", redis_url="redis://two:6379/0")
    assert second is first
    assert second.service_name == "first"
    assert second.redis_url == "redis://one:6379/0"


def test_two_mixin_combinations_are_two_classes_and_two_objects() -> None:
    redis = settings_class(RedisSettingsMixin)(service_name="a")
    sql = settings_class(SQLSettingsMixin)(service_name="b")
    assert redis is not sql
    assert type(redis) is not type(sql)


def test_core_settings_requires_a_service_name() -> None:
    with pytest.raises(ValidationError) as caught:
        settings_class()()
    assert "service_name" in str(caught.value)


def test_core_settings_defaults_an_environment_and_a_log_level() -> None:
    settings = settings_class()(service_name="svc")
    assert settings.environment == "local"
    assert settings.log_level == "INFO"


def test_core_settings_reads_its_own_fields_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_NAME", "from-env")
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = settings_class()()
    assert (settings.service_name, settings.environment, settings.log_level) == (
        "from-env",
        "staging",
        "DEBUG",
    )


def test_unknown_environment_variables_are_ignored_rather_than_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOMETHING_NOBODY_DECLARED", "1")
    settings = settings_class()(service_name="svc")
    assert not hasattr(settings, "something_nobody_declared")


def test_a_composed_settings_object_is_a_core_settings() -> None:
    settings = settings_class(RedisSettingsMixin, SQLSettingsMixin)(service_name="svc")
    assert isinstance(settings, CoreSettings)
    assert isinstance(settings, RedisSettingsMixin)
    assert isinstance(settings, SQLSettingsMixin)


def test_make_settings_registers_what_it_builds(make_settings: Any) -> None:
    built = make_settings(RedisSettingsMixin, redis_url="redis://x:6379/0")
    assert get_settings() is built
