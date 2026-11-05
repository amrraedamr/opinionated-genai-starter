"""What an OAuth service reads, one module per provider."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsOAuthConfig(Protocol):
    """Satisfied by `OAuthSettingsMixin`."""

    oauth_providers: str
    oauth_success_redirect: str
    oauth_state_ttl_seconds: int
    oauth_exchange_code_ttl_seconds: int
