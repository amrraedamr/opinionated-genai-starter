"""Who the current request or event is acting for: its tenant, its user and their bearer."""

import contextvars
from collections.abc import Generator
from contextlib import contextmanager

from base.models.definitions.tenancy import TenantId
from base.models.view_models.tokens import ForwardedBearer

current_tenant: contextvars.ContextVar[TenantId | None] = contextvars.ContextVar(
    "current_tenant", default=None
)
current_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_user", default=None
)
current_bearer: contextvars.ContextVar[ForwardedBearer | None] = contextvars.ContextVar(
    "current_bearer", default=None
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


class BearerContext:
    """Read and scope the token the caller presented, so a downstream call can act as them."""

    @staticmethod
    def get() -> ForwardedBearer | None:
        """The caller's token, or None outside a request that presented one."""
        return current_bearer.get()

    @staticmethod
    def set(token: ForwardedBearer | None) -> None:
        """Present `token` downstream until the current context ends."""
        current_bearer.set(token)

    @staticmethod
    @contextmanager
    def scope(token: ForwardedBearer | None) -> Generator[None]:
        """Present `token` downstream inside the block, restoring the previous one after."""
        previous = current_bearer.set(token)
        try:
            yield
        finally:
            current_bearer.reset(previous)


__all__ = [
    "BearerContext",
    "CallerContext",
    "TenantContext",
    "current_bearer",
    "current_tenant",
    "current_user",
]
