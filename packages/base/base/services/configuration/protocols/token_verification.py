from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsTokenVerificationConfig(Protocol):
    """Satisfied by `TokenVerificationMixin`."""

    auth_public_key_path: str
