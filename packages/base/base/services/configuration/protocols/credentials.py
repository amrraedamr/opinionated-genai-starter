from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsCredentialsConfig(Protocol):
    """Satisfied by `CredentialsSettingsMixin`."""

    outbound_token_url: str
    outbound_client_id: str
    outbound_client_secret: str
    outbound_scope: str
    outbound_credential_placement: str
    outbound_token_refresh_skew_seconds: int
    outbound_verify_tls: bool
