from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsIdpConfig(Protocol):
    """Satisfied by `IdpResourceServerMixin`."""

    idp_resource_server_enabled: bool
    idp_issuer: str
    idp_audience: str
    idp_jwks_url: str
    idp_discovery_url: str
    idp_algorithms: str
    idp_email_claim: str
