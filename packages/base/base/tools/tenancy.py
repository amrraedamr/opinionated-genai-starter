"""The tenant the current request or event is acting for."""

import contextvars
from collections.abc import Generator
from contextlib import contextmanager

from base.models.definitions.tenancy import TenantId

current_tenant: contextvars.ContextVar[TenantId | None] = contextvars.ContextVar(
    "current_tenant", default=None
)


class TenantContext:
    """Read and scope the tenant a unit of work is acting for."""

    @staticmethod
    def get() -> TenantId | None:
        """The tenant in effect, or None when nothing set one."""
        return current_tenant.get()

    @staticmethod
    @contextmanager
    def scope(tenant_id: TenantId | None) -> Generator[None]:
        """Act for `tenant_id` inside the block, restoring the previous one after."""
        token = current_tenant.set(tenant_id)
        try:
            yield
        finally:
            current_tenant.reset(token)


__all__ = ["TenantContext", "current_tenant"]
