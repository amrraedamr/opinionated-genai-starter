"""Which customer a request, a row or an event belongs to."""

from typing import NewType

TenantId = NewType("TenantId", str)

DEFAULT_TENANT_CLAIM = "tid"
"""The token claim a tenant is read from, named as Entra ID names it."""

RLS_SETTING = "app.tenant_id"
"""The Postgres run-time parameter a row-level policy compares against."""

RLS_POLICY_NAME = "tenant_isolation"
