"""Who the current request or event is acting for: its tenant and its user."""

import contextvars
from collections.abc import Generator
from contextlib import contextmanager

from base.models.definitions.tenancy import TenantId

current_tenant: contextvars.ContextVar[TenantId | None] = contextvars.ContextVar(
    "current_tenant", default=None
)
current_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_user", default=None
)


class TenantContext:
    """Read and scope the tenant a unit of work is acting for."""

    @staticmethod
    def get() -> TenantId | None:
        """The tenant in effect, or None when nothing set one."""
        return current_tenant.get()

    @staticmethod
    def set(tenant_id: TenantId | None) -> None:
        """Act for `tenant_id` until the current context ends."""
        current_tenant.set(tenant_id)

    @staticmethod
    @contextmanager
    def scope(tenant_id: TenantId | None) -> Generator[None]:
        """Act for `tenant_id` inside the block, restoring the previous one after."""
        token = current_tenant.set(tenant_id)
        try:
            yield
        finally:
            current_tenant.reset(token)


class CallerContext:
    """Read and scope the user a unit of work is acting for, so budgets can be per user."""

    @staticmethod
    def get() -> str | None:
        """The user in effect, or None when nothing set one."""
        return current_user.get()

    @staticmethod
    def set(user: str | None) -> None:
        """Act for `user` until the current context ends."""
        current_user.set(user)

    @staticmethod
    @contextmanager
    def scope(user: str | None) -> Generator[None]:
        """Act for `user` inside the block, restoring the previous one after."""
        token = current_user.set(user)
        try:
            yield
        finally:
            current_user.reset(token)


__all__ = ["CallerContext", "TenantContext", "current_tenant", "current_user"]
