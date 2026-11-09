"""Row-level security against a real Postgres, where the fence is the point.

`SQLStore` narrows the reads it issues, so a test driving the store proves
nothing here — it would pass with the policy switched off. These go the other
way deliberately: they read through the ORM **without** the store, which is the
path `PgVectorService` takes and the one carrying no tenant predicate.

The table is a declarative model using the real `TenantScopedMixin`, so the
column the policy compares against is the one a scoped model actually gets.

Everything runs under a deliberately unprivileged role. The compose role is a
superuser with `BYPASSRLS`, so a test using it would read every tenant's rows
and report that as success.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from sqlalchemy import Integer, String, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column

from base.models.definitions.tenancy import RLS_SETTING, TenantId
from base.models.orm import TenantScopedMixin
from base.services.configuration.mixins import (
    PostgresSettingsMixin,
    SQLSettingsMixin,
    TenancySettingsMixin,
)
from base.services.sql.rls import RowLevelSecurity
from base.services.sql.service import SQLService
from base.tools.tenancy import TenantContext

pytestmark = pytest.mark.integration

ACME = TenantId("acme")
GLOBEX = TenantId("globex")
ROLE = "rls_tenant_tester"


class Base(MappedAsDataclass, DeclarativeBase):
    pass


class Note(TenantScopedMixin, Base):
    """A scoped row, carrying exactly what a real scoped model carries."""

    __tablename__ = "rls_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    body: Mapped[str] = mapped_column(String(200))


def configure(make_settings: Any, *, rls: bool) -> Any:
    return make_settings(
        SQLSettingsMixin,
        PostgresSettingsMixin,
        TenancySettingsMixin,
        sql_backend="postgresql",
        pg_host="localhost",
        pg_port="5432",
        pg_database="postgres",
        pg_username="local",
        pg_password="local",
        db_ssl_enabled=False,
        tenancy_enabled=True,
        tenancy_rls_enabled=rls,
    )


@pytest.fixture
async def notes(make_settings: Any) -> AsyncIterator[SQLService]:
    """A fenced table with rows for two tenants, dropped afterwards."""
    configure(make_settings, rls=True)
    service = SQLService()

    async with service.atransaction() as session:
        connection = await session.connection()
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
        for statement in RowLevelSecurity.statements(Note.__tablename__):
            await session.execute(statement)
        await session.execute(
            text(
                f"DO $$ BEGIN CREATE ROLE {ROLE} NOLOGIN; "
                "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        )
        await session.execute(text(f"GRANT ALL ON {Note.__tablename__} TO {ROLE}"))
        await session.execute(
            Note.__table__.insert(),
            [
                {"id": 1, "tenant_id": ACME, "body": "acme one"},
                {"id": 2, "tenant_id": ACME, "body": "acme two"},
                {"id": 3, "tenant_id": GLOBEX, "body": "globex one"},
            ],
        )

    try:
        yield service
    finally:
        async with service.atransaction() as session:
            connection = await session.connection()
            await connection.run_sync(Base.metadata.drop_all)
        await service.adispose()


async def ids_visible(service: SQLService, tenant: TenantId | None) -> list[int]:
    """Ids the ORM can see, as an unprivileged role, acting for `tenant`."""
    with TenantContext.scope(tenant):
        async with service.atransaction() as session:
            await session.execute(text(f"SET LOCAL ROLE {ROLE}"))
            result = await session.execute(select(Note.id).order_by(Note.id))
            return list(result.scalars())


async def test_a_read_sees_only_the_current_tenant(notes: SQLService) -> None:
    """The whole point: no store involved, and the other tenant is not there."""
    assert await ids_visible(notes, ACME) == [1, 2]


async def test_the_other_tenant_sees_its_own_rows(notes: SQLService) -> None:
    assert await ids_visible(notes, GLOBEX) == [3]


async def test_no_tenant_sees_nothing(notes: SQLService) -> None:
    """Failing closed: an unset parameter must not mean unrestricted."""
    assert await ids_visible(notes, None) == []


async def test_the_pgvector_path_is_fenced_too(notes: SQLService) -> None:
    """`PgVectorService` opens sessions through `_aget_session`, not a transaction.

    That is the path with no tenant predicate anywhere in it, and the reason
    this feature exists, so it is pinned separately.
    """
    with TenantContext.scope(ACME):
        async with notes._aget_session() as session:
            await session.execute(text(f"SET LOCAL ROLE {ROLE}"))
            result = await session.execute(select(Note.id).order_by(Note.id))
            assert list(result.scalars()) == [1, 2]


async def test_a_write_cannot_land_in_another_tenant(notes: SQLService) -> None:
    """`WITH CHECK`: writing across the fence is refused, not silently lost."""
    with TenantContext.scope(ACME):
        with pytest.raises(Exception, match="row-level security|policy"):
            async with notes.atransaction() as session:
                await session.execute(text(f"SET LOCAL ROLE {ROLE}"))
                await session.execute(
                    Note.__table__.insert(),
                    [{"id": 9, "tenant_id": GLOBEX, "body": "smuggled"}],
                )


async def test_a_delete_cannot_reach_another_tenant(notes: SQLService) -> None:
    with TenantContext.scope(ACME):
        async with notes.atransaction() as session:
            await session.execute(text(f"SET LOCAL ROLE {ROLE}"))
            await session.execute(Note.__table__.delete())
    assert await ids_visible(notes, GLOBEX) == [3]


async def test_the_parameter_is_set_on_every_transaction(
    notes: SQLService,
) -> None:
    """A second transaction must be fenced too, not only the first."""
    with TenantContext.scope(ACME):
        for _ in range(2):
            async with notes.atransaction() as session:
                result = await session.execute(
                    select(text(f"current_setting('{RLS_SETTING}', true)"))
                )
                assert result.scalar() == ACME


async def test_the_tenant_does_not_leak_between_transactions(
    notes: SQLService,
) -> None:
    """`set_config(..., true)` is transaction-scoped, so it must not persist."""
    with TenantContext.scope(ACME):
        async with notes.atransaction() as session:
            await session.execute(select(text("1")))

    async with notes.atransaction() as session:
        result = await session.execute(
            select(text(f"current_setting('{RLS_SETTING}', true)"))
        )
        assert result.scalar() in ("", None)


async def test_disabled_changes_nothing(make_settings: Any) -> None:
    """Opt-in means a deployment that never asked sees no behaviour change."""
    configure(make_settings, rls=False)
    service = SQLService()
    async with service.atransaction() as session:
        connection = await session.connection()
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
        await session.execute(
            Note.__table__.insert(),
            [
                {"id": 1, "tenant_id": ACME, "body": "one"},
                {"id": 2, "tenant_id": GLOBEX, "body": "two"},
            ],
        )
    try:
        with TenantContext.scope(ACME):
            async with service.atransaction() as session:
                parameter = await session.execute(
                    select(text(f"current_setting('{RLS_SETTING}', true)"))
                )
                assert parameter.scalar() in ("", None), "the hook ran while off"
                rows = await session.execute(select(Note.id))
                assert len(list(rows.scalars())) == 2
    finally:
        async with service.atransaction() as session:
            connection = await session.connection()
            await connection.run_sync(Base.metadata.drop_all)
        await service.adispose()
