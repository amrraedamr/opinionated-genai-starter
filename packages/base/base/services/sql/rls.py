"""Fencing a tenant's rows at the database with Postgres row-level security."""

from typing import TYPE_CHECKING, ClassVar

from base.models.definitions.tenancy import RLS_POLICY_NAME, RLS_SETTING
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsTenancyConfig
from base.tools.errors import MissingExtraError
from base.tools.tenancy import TenantContext

try:
    from sqlalchemy import event, func, select
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import Session
    from sqlalchemy.schema import DDLElement
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="RowLevelSecurity") from exc

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.orm import SessionTransaction
    from sqlalchemy.sql.compiler import DDLCompiler


class _TableDDL(DDLElement):
    """DDL naming one table, which is all these have in common."""

    inherit_cache = True

    def __init__(self, table: str) -> None:
        self.table = table


class EnableRowLevelSecurity(_TableDDL):
    """Turn the feature on for a table. On its own it exempts the owner."""


class ForceRowLevelSecurity(_TableDDL):
    """Apply the policy to the table's owner too, which a service usually is."""


class DisableRowLevelSecurity(_TableDDL):
    """Turn it back off, for a migration coming down."""


class UnforceRowLevelSecurity(_TableDDL):
    """Exempt the owner again, for a migration coming down."""


class DropTenantPolicy(_TableDDL):
    """Remove the policy, tolerating its absence."""


class CreateTenantPolicy(DDLElement):
    """Confine every row to the tenant named by the run-time parameter."""

    inherit_cache = True

    def __init__(self, table: str, column: str = "tenant_id") -> None:
        self.table = table
        self.column = column


@compiles(EnableRowLevelSecurity, "postgresql")
def _compile_enable(
    element: EnableRowLevelSecurity, compiler: "DDLCompiler", **kw: object
) -> str:
    del kw
    quoted = compiler.preparer.quote(element.table)
    return f"ALTER TABLE {quoted} ENABLE ROW LEVEL SECURITY"


@compiles(ForceRowLevelSecurity, "postgresql")
def _compile_force(
    element: ForceRowLevelSecurity, compiler: "DDLCompiler", **kw: object
) -> str:
    del kw
    quoted = compiler.preparer.quote(element.table)
    return f"ALTER TABLE {quoted} FORCE ROW LEVEL SECURITY"


@compiles(DisableRowLevelSecurity, "postgresql")
def _compile_disable(
    element: DisableRowLevelSecurity, compiler: "DDLCompiler", **kw: object
) -> str:
    del kw
    quoted = compiler.preparer.quote(element.table)
    return f"ALTER TABLE {quoted} DISABLE ROW LEVEL SECURITY"


@compiles(UnforceRowLevelSecurity, "postgresql")
def _compile_unforce(
    element: UnforceRowLevelSecurity, compiler: "DDLCompiler", **kw: object
) -> str:
    del kw
    quoted = compiler.preparer.quote(element.table)
    return f"ALTER TABLE {quoted} NO FORCE ROW LEVEL SECURITY"


@compiles(CreateTenantPolicy, "postgresql")
def _compile_create_policy(
    element: CreateTenantPolicy, compiler: "DDLCompiler", **kw: object
) -> str:
    """`WITH CHECK` as well as `USING`, or a write could cross the fence."""
    del kw
    quote = compiler.preparer.quote
    predicate = f"{quote(element.column)} = current_setting('{RLS_SETTING}', true)"
    return (
        f"CREATE POLICY {quote(RLS_POLICY_NAME)} ON {quote(element.table)} "
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


@compiles(DropTenantPolicy, "postgresql")
def _compile_drop_policy(
    element: DropTenantPolicy, compiler: "DDLCompiler", **kw: object
) -> str:
    del kw
    quote = compiler.preparer.quote
    return f"DROP POLICY IF EXISTS {quote(RLS_POLICY_NAME)} ON {quote(element.table)}"


class RowLevelSecurity:
    """Puts the current tenant on every transaction, and writes the policy.

    Superusers and roles with `BYPASSRLS` are exempt however the policy is written.
    """

    _installed: ClassVar[bool] = False

    @classmethod
    def install(cls) -> None:
        """Hook every transaction start, once per process, since the tenant parameter is transaction-scoped."""
        if cls._installed:
            return
        event.listen(Session, "after_begin", cls._apply)
        cls._installed = True

    @staticmethod
    def _apply(
        session: Session,
        transaction: "SessionTransaction",
        connection: "Connection",
    ) -> None:
        """Tell this transaction its tenant via `set_config`, which unlike `SET LOCAL` takes bind parameters."""
        del session, transaction
        if connection.dialect.name != "postgresql":
            return
        if not RowLevelSecurity.enabled():
            return
        connection.execute(
            select(func.set_config(RLS_SETTING, TenantContext.get() or "", True))
        )

    @staticmethod
    def enabled() -> bool:
        """Whether a deployment asked for this. Off unless it did."""
        try:
            return settings_for(SupportsTenancyConfig).tenancy_rls_enabled
        except Exception:
            return False

    @staticmethod
    def statements(table: str, column: str = "tenant_id") -> list[DDLElement]:
        """The DDL that fences one table, for a migration to execute."""
        return [
            EnableRowLevelSecurity(table),
            ForceRowLevelSecurity(table),
            CreateTenantPolicy(table, column),
        ]

    @staticmethod
    def drop_statements(table: str) -> list[DDLElement]:
        """The reverse, for a migration that has to come back down."""
        return [
            DropTenantPolicy(table),
            UnforceRowLevelSecurity(table),
            DisableRowLevelSecurity(table),
        ]


__all__ = [
    "CreateTenantPolicy",
    "DisableRowLevelSecurity",
    "DropTenantPolicy",
    "EnableRowLevelSecurity",
    "ForceRowLevelSecurity",
    "RowLevelSecurity",
    "UnforceRowLevelSecurity",
]
