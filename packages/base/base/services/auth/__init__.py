"""Authentication mechanism: signing, verifying, bearer scheme, OAuth protocol."""

from base.services.auth.bearer import (
    current_principal,
    http_bearer,
    http_bearer_dependency,
    principal_dependency,
)
from base.services.auth.models import (
    AccessToken,
    Principal,
    RefreshToken,
    TokenPair,
)
from base.services.auth.tokens import TokenIssuer, TokenVerifier
from base.services.configuration.protocols import (
    SupportsIdpConfig,
    SupportsTokenIssuingConfig,
    SupportsTokenVerificationConfig,
)

__all__ = [
    "AccessToken",
    "Principal",
    "RefreshToken",
    "SupportsIdpConfig",
    "SupportsTokenIssuingConfig",
    "SupportsTokenVerificationConfig",
    "TokenIssuer",
    "TokenPair",
    "TokenVerifier",
    "current_principal",
    "http_bearer",
    "http_bearer_dependency",
    "principal_dependency",
]
