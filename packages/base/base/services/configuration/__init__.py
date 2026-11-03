"""Settings provider for `base` subsystems."""

from base.services.configuration.core import CoreSettings

_settings: CoreSettings | None = None


def configure(instance: CoreSettings) -> CoreSettings:
    """Register the service's settings. Call once, from the composition root."""
    global _settings
    _settings = instance
    return instance


def get_settings() -> CoreSettings:
    if _settings is None:
        raise RuntimeError(
            "Settings not configured. Call "
            "base.services.configuration.configure(YourSettings()) in your "
            "service entrypoint before using any base service."
        )
    return _settings


def is_configured() -> bool:
    return _settings is not None


def reset() -> None:
    """Drop the registered settings. Intended for tests."""
    global _settings
    _settings = None


__all__ = ["CoreSettings", "configure", "get_settings", "is_configured", "reset"]
