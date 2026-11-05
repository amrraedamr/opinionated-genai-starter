"""Settings provider for `base` subsystems."""

from typing import cast

from base.services.configuration.core import CoreSettings
from base.tools.errors import MissingSettingsError

_settings: CoreSettings | None = None

_checked: set[tuple[int, int]] = set()
"""Which settings object has already been checked against which contract.

The settings are configured once, so the check is worth doing on first read
and not on every one.
"""


def configure(instance: CoreSettings) -> CoreSettings:
    """Register the service's settings. Call once, from the composition root."""
    global _settings
    _settings = instance
    _checked.clear()
    return instance


def get_settings() -> CoreSettings:
    if _settings is None:
        raise RuntimeError(
            "Settings not configured. Call "
            "base.services.configuration.configure(YourSettings()) in your "
            "service entrypoint before using any base service."
        )
    return _settings


def settings_for[T](contract: type[T]) -> T:
    """The service's settings, read as the contract a subsystem declares.

    A subsystem states what it needs with a protocol and reads the settings
    through it, so the fields it touches are checked and the ones it does not
    stay invisible to it. The settings satisfy that protocol structurally
    rather than by inheritance, which is why this casts.

    The contract is also checked, once per settings object. A subsystem
    reading a mixin the service never composed is a composition mistake, and
    saying which contract went unmet beats an AttributeError raised wherever
    the first missing field happened to be read.
    """
    settings = get_settings()
    key = (id(settings), id(contract))
    if key not in _checked:
        if isinstance(contract, type) and not isinstance(settings, contract):
            raise MissingSettingsError(type(settings).__name__, contract.__name__)
        _checked.add(key)
    return cast(T, settings)


def is_configured() -> bool:
    return _settings is not None


def reset() -> None:
    """Drop the registered settings. Intended for tests."""
    _checked.clear()
    global _settings
    _settings = None


__all__ = ["CoreSettings", "configure", "get_settings", "is_configured", "reset"]
