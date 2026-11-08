from pydantic import BaseModel

from base.models.definitions.tenancy import DEFAULT_TENANT_CLAIM


class TenancySettingsMixin(BaseModel):
    """Separate customers' data. Off unless a service turns it on."""

    tenancy_enabled: bool = False
    tenant_claim: str = DEFAULT_TENANT_CLAIM
    tenancy_rls_enabled: bool = False
