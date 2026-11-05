from pydantic import BaseModel


class IdpSettingsMixin(BaseModel):
    """Accept enterprise IdP access tokens on this API."""

    idp_resource_server_enabled: bool = False
    idp_issuer: str = ""
    idp_audience: str = ""
    idp_jwks_url: str = ""
    idp_discovery_url: str = ""
    idp_algorithms: str = "RS256"
    idp_provider_name: str = "entra"
    idp_email_claim: str = "email"
