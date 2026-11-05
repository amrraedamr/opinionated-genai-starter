"""Login-with-X settings, one module per provider.

This holds what no provider owns. A service composes it plus a mixin from
beside it for each provider it offers, so offering Google does not also
declare Apple's signing key.
"""

from pydantic import BaseModel


class OAuthSettingsMixin(BaseModel):
    """Login-with-X, in terms no provider owns.

    `oauth_providers` is a comma-separated list of enabled names. A service
    composes this plus a mixin per provider it actually offers, so declaring
    Google does not also declare Apple's signing key.
    """

    oauth_providers: str = ""
    oauth_success_redirect: str = "http://localhost:3000/auth/callback"
    oauth_state_ttl_seconds: int = 600
    oauth_exchange_code_ttl_seconds: int = 60
