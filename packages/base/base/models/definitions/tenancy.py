"""Which customer a request, a row or an event belongs to."""

from typing import NewType

TenantId = NewType("TenantId", str)

DEFAULT_TENANT_CLAIM = "tid"
"""The claim a tenant is read from: Entra ID names it `tid`, so a first-party
token uses the same name rather than inventing one."""
