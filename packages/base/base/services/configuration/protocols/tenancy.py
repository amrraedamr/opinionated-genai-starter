from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsTenancyConfig(Protocol):
    """Satisfied by `TenancySettingsMixin`."""

    tenancy_enabled: bool
    tenant_claim: str
    tenancy_rls_enabled: bool
