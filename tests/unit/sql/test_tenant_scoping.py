"""Separating one customer's rows from another's.

The property under test is not that a tenant sees its own rows: it is that it
cannot reach anyone else's by any route a store offers, and that a store with
no tenant in context refuses rather than acting for nobody. A scoped query
without a tenant would read `tenant_id IS NULL` and answer with an empty list,
which looks like "no rows" rather than the mistake it is.
"""

import pathlib
from typing import Any, ClassVar

import pytest
from sqlalchemy import Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column

from base.models.definitions.errors import DatabaseErrors, TenancyErrors
from base.models.definitions.tenancy import TenantId
from base.models.orm import SerializableMixin, TenantScopedMixin, TimestampMixin
from base.models.view_models.database import FilteredPaginationParams
from base.services.sql.service import SQLService
from base.services.store.sql import SQLStore
from base.tools.errors import ErrorsException
from base.tools.tenancy import TenantContext

ACME = TenantId("acme")
GLOBEX = TenantId("globex")


def build_note_model(table: str) -> type[Any]:
    """A tenant-scoped, soft-deleted table on a registry of its own."""

    class Base(MappedAsDataclass, DeclarativeBase, SerializableMixin):
        pass

    class Note(TimestampMixin, TenantScopedMixin, Base):
        __tablename__ = table

        id: Mapped[int] = mapped_column(
            Integer, primary_key=True, autoincrement=True, init=False
        )
        title: Mapped[str] = mapped_column(String(200))
        body: Mapped[str | None] = mapped_column(String(500), default=None)

    return Note


class NoteStore(SQLStore[Any]):
    search_fields: ClassVar[tuple[str, ...]] = ("title", "body")


@pytest.fixture
def note(sqlite_settings: Any, unique: str) -> type[Any]:
    model = build_note_model(f"note_{unique}")
    with SQLService()._get_session() as session:
        model.metadata.create_all(session.get_bind())
    return model


@pytest.fixture
def notes(note: type[Any]) -> NoteStore:
    return NoteStore(note)


@pytest.fixture
def seeded(notes: NoteStore, note: type[Any]) -> None:
    """One row for each of two tenants."""
    with TenantContext.scope(ACME):
        notes.add(note(title="acme-one"))
    with TenantContext.scope(GLOBEX):
        notes.add(note(title="globex-one"))


# Opting in


def test_a_model_without_the_mixin_is_not_scoped(store: Any) -> None:
    assert store._tenant_scoped is False


def test_a_model_carrying_the_mixin_is_scoped(notes: NoteStore) -> None:
    assert notes._tenant_scoped is True


def test_an_unscoped_store_needs_no_tenant_at_all(
    store: Any, widget: type[Any]
) -> None:
    """A service that did not opt in pays nothing: no column, no filter, no claim."""
    store.add(widget(name="free"))
    assert len(store.get_all()) == 1


# Writes


def test_a_row_is_stamped_with_the_tenant_that_wrote_it(
    notes: NoteStore, note: type[Any]
) -> None:
    with TenantContext.scope(ACME):
        created = notes.add(note(title="one"))
        assert created.tenant_id == ACME


def test_many_rows_are_all_stamped(notes: NoteStore, note: type[Any]) -> None:
    with TenantContext.scope(ACME):
        created = notes.add_many([note(title="a"), note(title="b")])
    assert [row.tenant_id for row in created] == [ACME, ACME]


async def test_the_async_write_stamps_too(notes: NoteStore, note: type[Any]) -> None:
    with TenantContext.scope(ACME):
        created = await notes.aadd(note(title="one"))
        assert created.tenant_id == ACME


# Reads are scoped


def test_a_tenant_sees_only_its_own_rows(notes: NoteStore, seeded: None) -> None:
    with TenantContext.scope(ACME):
        assert [row.title for row in notes.get_all()] == ["acme-one"]
    with TenantContext.scope(GLOBEX):
        assert [row.title for row in notes.get_all()] == ["globex-one"]


def test_another_tenants_row_reads_as_not_found_rather_than_forbidden(
    notes: NoteStore, note: type[Any]
) -> None:
    """Not found, not refused: a refusal would confirm the row exists."""
    with TenantContext.scope(ACME):
        theirs = notes.add(note(title="secret"))
    with TenantContext.scope(GLOBEX):
        with pytest.raises(ErrorsException) as caught:
            notes.get(theirs.id)
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND


def test_another_tenants_rows_are_not_reachable_in_bulk(
    notes: NoteStore, note: type[Any]
) -> None:
    with TenantContext.scope(ACME):
        theirs = notes.add(note(title="secret"))
    with TenantContext.scope(GLOBEX):
        assert notes.get_many([theirs.id]) == []


def test_a_paged_read_is_scoped(notes: NoteStore, seeded: None) -> None:
    params = FilteredPaginationParams()
    with TenantContext.scope(ACME):
        page = notes.get_all_paged(params)
    assert [row.title for row in page.data] == ["acme-one"]
    assert page.total == 1


def test_a_search_cannot_reach_across_tenants(
    notes: NoteStore, note: type[Any]
) -> None:
    with TenantContext.scope(ACME):
        notes.add(note(title="findme-acme"))
    with TenantContext.scope(GLOBEX):
        page = notes.get_all_paged(FilteredPaginationParams(search="findme"))
    assert page.data == []


async def test_the_async_read_is_scoped(notes: NoteStore, seeded: None) -> None:
    with TenantContext.scope(GLOBEX):
        rows = await notes.aget_all()
    assert [row.title for row in rows] == ["globex-one"]


# Writes cannot cross either


def test_another_tenants_row_cannot_be_edited(
    notes: NoteStore, note: type[Any]
) -> None:
    with TenantContext.scope(ACME):
        theirs = notes.add(note(title="theirs"))
    with TenantContext.scope(GLOBEX):
        assert notes.edit(theirs.id, note(title="hijacked")) is None
    with TenantContext.scope(ACME):
        assert notes.get(theirs.id).title == "theirs"


def test_another_tenants_row_cannot_be_deleted(
    notes: NoteStore, note: type[Any]
) -> None:
    with TenantContext.scope(ACME):
        theirs = notes.add(note(title="theirs"))
    with TenantContext.scope(GLOBEX):
        assert notes.delete(theirs.id) is False
    with TenantContext.scope(ACME):
        assert notes.get(theirs.id) is not None


def test_an_edit_cannot_move_a_row_into_another_tenant(
    notes: NoteStore, note: type[Any]
) -> None:
    """`tenant_id` is protected, so a patch naming one is ignored like an id."""
    with TenantContext.scope(ACME):
        mine = notes.add(note(title="mine"))
        patch = note(title="renamed")
        patch.tenant_id = GLOBEX
        edited = notes.edit(mine.id, patch)
        assert edited.tenant_id == ACME
        assert edited.title == "renamed"


def test_the_tenant_column_is_protected_from_edits() -> None:
    assert "tenant_id" in SQLStore.protected_fields


# Acting for nobody is refused


def test_a_scoped_read_with_no_tenant_refuses(notes: NoteStore) -> None:
    with pytest.raises(ErrorsException) as caught:
        notes.get_all()
    assert caught.value.error_enum is TenancyErrors.TENANT_NOT_IN_CONTEXT


def test_a_scoped_read_refuses_rather_than_answering_with_nothing(
    notes: NoteStore, seeded: None
) -> None:
    """The failure this prevents: `tenant_id IS NULL` reads as an empty result."""
    with pytest.raises(ErrorsException):
        notes.get_all()


def test_a_scoped_write_with_no_tenant_refuses(
    notes: NoteStore, note: type[Any]
) -> None:
    with pytest.raises(ErrorsException):
        notes.add(note(title="orphan"))


def test_a_scoped_get_by_id_with_no_tenant_refuses(notes: NoteStore) -> None:
    with pytest.raises(ErrorsException):
        notes.get(1)


def test_a_scoped_delete_with_no_tenant_refuses(notes: NoteStore) -> None:
    with pytest.raises(ErrorsException):
        notes.delete(1)


async def test_a_scoped_async_read_with_no_tenant_refuses(notes: NoteStore) -> None:
    with pytest.raises(ErrorsException):
        await notes.aget_all()


def test_an_explicitly_cleared_tenant_is_still_no_tenant(notes: NoteStore) -> None:
    with TenantContext.scope(ACME):
        with TenantContext.scope(None):
            with pytest.raises(ErrorsException):
                notes.get_all()


# The column itself


def test_the_tenant_column_is_indexed_because_every_read_filters_on_it(
    note: type[Any],
) -> None:
    indexed = {
        column.name for index in note.__table__.indexes for column in index.columns
    }
    assert "tenant_id" in indexed


def test_the_tenant_column_is_not_a_constructor_argument(note: type[Any]) -> None:
    """A caller never sets it: the store stamps it from context."""
    row = note(title="one")
    assert not hasattr(row, "tenant_id") or row.tenant_id is None


def test_soft_deleted_rows_stay_invisible_within_a_tenant(
    notes: NoteStore, note: type[Any]
) -> None:
    with TenantContext.scope(ACME):
        row = notes.add(note(title="gone"))
        notes.delete(row.id)
        assert notes.get_all() == []


def test_the_sqlite_file_really_holds_both_tenants_rows(
    notes: NoteStore, note: type[Any], sqlite_path: pathlib.Path
) -> None:
    """Scoping is a filter, not a separate table: both rows are in one table."""
    with TenantContext.scope(ACME):
        notes.add(note(title="a"))
    with TenantContext.scope(GLOBEX):
        notes.add(note(title="b"))
    with SQLService()._get_session() as session:
        from sqlalchemy import text

        table = notes._model.__tablename__
        rows = session.execute(text(f"SELECT tenant_id FROM {table}")).scalars().all()
    assert sorted(rows) == ["acme", "globex"]
