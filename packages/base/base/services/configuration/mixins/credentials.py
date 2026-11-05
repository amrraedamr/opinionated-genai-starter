from typing import Literal

from pydantic import BaseModel


class CredentialsSettingsMixin(BaseModel):
    """Outbound OAuth2 credentials — how this service authenticates itself.

    The inbound half is `TokenVerificationSettingsMixin` / `OAuthSettingsMixin`. One
    mixin covers every authorization server; nothing is defaulted to a vendor.
    """

    outbound_token_url: str = ""
    outbound_client_id: str = ""
    outbound_client_secret: str = ""
    outbound_scope: str = ""
    outbound_credential_placement: Literal["basic", "form"] = "basic"
    outbound_token_refresh_skew_seconds: int = 60
    outbound_verify_tls: bool = True
