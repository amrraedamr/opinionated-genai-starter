from typing import Protocol, runtime_checkable

from base.services.configuration.protocols.token_verification import (
    SupportsTokenVerificationConfig,
)


@runtime_checkable
class SupportsTokenIssuingConfig(SupportsTokenVerificationConfig, Protocol):
    """Satisfied by `TokenIssuingSettingsMixin`."""

    auth_private_key_path: str
    access_token_expiry_seconds: int
    refresh_token_expiry_seconds: int
