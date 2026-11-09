"""The DDL that fences a table, and the switch that decides whether it runs.

The statements are compiled for the Postgres dialect and asserted as text,
because two words in them carry the whole feature. Without `FORCE` the table's
owner is exempt, and a service usually connects as the owner. Without
`WITH CHECK` a tenant can write a row it will never be able to read back.
"""

from typing import Any

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import DDLElement

from base.models.definitions.tenancy import RLS_POLICY_NAME, RLS_SETTING, TenantId
from base.services.configuration.mixins import TenancySettingsMixin
from base.services.sql.rls import RowLevelSecurity
from base.tools.tenancy import TenantContext

DIALECT = postgresql.dialect()


def rendered(statements: list[DDLElement]) -> list[str]:
    """What Postgres would actually receive."""
    return [str(statement.compile(dialect=DIALECT)).strip() for statement in statements]


def policy_for(table: str = "documents", column: str = "tenant_id") -> str:
    return next(
        s for s in rendered(RowLevelSecurity.statements(table, column)) if "POLICY" in s
    )


def test_the_table_is_told_to_force_the_policy() -> None:
    """Enabling alone leaves the owner exempt, which is usually the service."""
    assert any(
        "FORCE ROW LEVEL SECURITY" in s
        for s in rendered(RowLevelSecurity.statements("documents"))
    )


def test_the_policy_is_enabled_before_it_is_forced() -> None:
    statements = rendered(RowLevelSecurity.statements("documents"))
    enable = next(i for i, s in enumerate(statements) if "ENABLE ROW" in s)
    force = next(i for i, s in enumerate(statements) if "FORCE ROW" in s)
    assert enable < force


def test_the_policy_covers_writes_as_well_as_reads() -> None:
    """`USING` alone would let a row be written into another tenant."""
    policy = policy_for()
    assert "USING" in policy
    assert "WITH CHECK" in policy


def test_the_policy_compares_against_the_run_time_parameter() -> None:
    policy = policy_for()
    assert f"current_setting('{RLS_SETTING}', true)" in policy
    assert "tenant_id =" in policy


def test_a_missing_parameter_is_tolerated_not_an_error() -> None:
    """The `true` is `missing_ok`: unset reads as NULL, which matches nothing."""
    assert ", true)" in policy_for()


def test_the_column_can_be_named_by_the_caller() -> None:
    assert "owner_id =" in policy_for(column="owner_id")


def test_an_awkward_identifier_is_quoted_by_the_dialect() -> None:
    """A table named for a reserved word must not produce broken DDL."""
    assert all('"user"' in s for s in rendered(RowLevelSecurity.statements("user")))
    assert '"order" =' in policy_for("user", "order")


def test_every_statement_names_the_table() -> None:
    assert all(
        "documents" in s for s in rendered(RowLevelSecurity.statements("documents"))
    )


def test_the_policy_is_named_the_same_on_every_table() -> None:
    for table in ("documents", "embeddings"):
        assert RLS_POLICY_NAME in policy_for(table)


def test_the_drop_reverses_what_was_applied() -> None:
    """A migration that cannot come back down is not a migration."""
    statements = rendered(RowLevelSecurity.drop_statements("documents"))
    assert any("DROP POLICY" in s for s in statements)
    assert any("NO FORCE" in s for s in statements)
    assert any("DISABLE ROW LEVEL SECURITY" in s for s in statements)


def test_dropping_a_policy_that_is_not_there_is_not_an_error() -> None:
    """A down-migration runs against databases that never ran the up."""
    assert any(
        "IF EXISTS" in s
        for s in rendered(RowLevelSecurity.drop_statements("documents"))
    )


def test_the_policy_is_dropped_before_the_table_is_unforced() -> None:
    assert rendered(RowLevelSecurity.drop_statements("documents"))[0].startswith(
        "DROP POLICY"
    )


def test_it_is_off_unless_a_deployment_asks(make_settings: Any) -> None:
    make_settings(TenancySettingsMixin)
    assert RowLevelSecurity.enabled() is False


def test_it_is_on_when_a_deployment_asks(make_settings: Any) -> None:
    make_settings(TenancySettingsMixin, tenancy_rls_enabled=True)
    assert RowLevelSecurity.enabled() is True


def test_unconfigured_settings_mean_off_rather_than_a_crash() -> None:
    """This runs on every transaction, including before anything is configured."""
    assert RowLevelSecurity.enabled() is False


def test_installing_twice_registers_one_listener() -> None:
    """`SQLClient` is keyed per backend, so install runs more than once."""
    RowLevelSecurity.install()
    RowLevelSecurity.install()
    assert RowLevelSecurity._installed is True


@pytest.mark.parametrize("tenant", [TenantId("acme"), None])
def test_the_hook_ignores_backends_without_row_level_security(
    make_settings: Any, tenant: TenantId | None
) -> None:
    """SQLite has no such thing, so the hook must leave it alone."""
    make_settings(TenancySettingsMixin, tenancy_rls_enabled=True)

    class Dialect:
        name = "sqlite"

    class Connection:
        dialect = Dialect()

        def execute(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("SQLite was sent a Postgres parameter")

    with TenantContext.scope(tenant):
        RowLevelSecurity._apply(None, None, Connection())  # type: ignore[arg-type]


def test_the_tenant_travels_as_a_bound_parameter(make_settings: Any) -> None:
    """A tenant pasted into SQL is how injection gets in, so it is bound."""
    make_settings(TenancySettingsMixin, tenancy_rls_enabled=True)
    captured: list[Any] = []

    class Dialect:
        name = "postgresql"

    class Connection:
        dialect = Dialect()

        def execute(self, statement: Any, *args: Any, **kwargs: Any) -> None:
            captured.append(statement)

    with TenantContext.scope(TenantId("'; DROP TABLE users; --")):
        RowLevelSecurity._apply(None, None, Connection())  # type: ignore[arg-type]

    assert len(captured) == 1
    assert "DROP TABLE" not in str(captured[0])
