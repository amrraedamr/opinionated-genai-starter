"""`AbstractSequenceStore`'s contract, honoured by a real SQL implementation.

The in-memory store proves the contract in one process. This proves it where
a sequence actually has to hold: across connections, with the database — not
a lock in this process — deciding who wins a collision. The store below is
the smallest implementation that satisfies the contract, written here so the
contract itself is what is under test.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import (
    Integer,
    Select,
    String,
    UniqueConstraint,
    Uuid,
    delete,
    func,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    mapped_column,
)

from base.models.view_models.exceptions.sql import SQLConflictError
from base.services.sql.service import SQLService
from base.services.store import AbstractSequenceStore
from base.tools.database import SQLResultFormats


@dataclass
class Note:
    """The stored item: anything with a `sequence` satisfies the contract."""

    sequence: int
    body: str


def build_note_model(table: str) -> type[Any]:
    """One row per note, with the sequence unique within its parent."""

    class Base(MappedAsDataclass, DeclarativeBase):
        pass

    class NoteDTO(Base):
        __tablename__ = table
        __table_args__ = (
            UniqueConstraint(
                "parent_id", "sequence", name=f"uq_{table}_parent_sequence"
            ),
        )

        id: Mapped[uuid.UUID] = mapped_column(
            Uuid, primary_key=True, insert_default=uuid.uuid4, init=False
        )
        parent_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
        sequence: Mapped[int] = mapped_column(Integer)
        body: Mapped[str] = mapped_column(String(200))

    return NoteDTO


class SqlNoteStore(AbstractSequenceStore[uuid.UUID, Note]):
    """The smallest implementation that satisfies `AbstractSequenceStore`."""

    def __init__(self, model: type[Any]) -> None:
        self._model: type[Any] = model
        self._sql: SQLService = SQLService()

    def _to_dto(self, parent_id: uuid.UUID, note: Note) -> Any:
        return self._model(parent_id=parent_id, sequence=note.sequence, body=note.body)

    @staticmethod
    def _from_dto(dto: Any) -> Note:
        return Note(sequence=dto.sequence, body=dto.body)

    def _ordered(self, parent_id: uuid.UUID, limit: int | None) -> Select[Any]:
        query = select(self._model).where(self._model.parent_id == parent_id)
        if limit is None:
            return query.order_by(self._model.sequence.asc())
        return query.order_by(self._model.sequence.desc()).limit(limit)

    @staticmethod
    def _ascending(rows: list[Any], limit: int | None) -> list[Any]:
        return rows if limit is None else list(reversed(rows))

    def _counting(self, parent_id: uuid.UUID) -> Select[Any]:
        return (
            select(func.count())
            .select_from(self._model)
            .where(self._model.parent_id == parent_id)
        )

    def _highest(self, parent_id: uuid.UUID) -> Select[Any]:
        return select(func.max(self._model.sequence)).where(
            self._model.parent_id == parent_id
        )

    def _append(self, parent_id: uuid.UUID, items: list[Note]) -> list[Note]:
        rows = self._sql.upsert_rows([self._to_dto(parent_id, i) for i in items])
        return [self._from_dto(row) for row in rows]

    async def _aappend(self, parent_id: uuid.UUID, items: list[Note]) -> list[Note]:
        rows = await self._sql.aupsert_rows([self._to_dto(parent_id, i) for i in items])
        return [self._from_dto(row) for row in rows]

    def _get(self, parent_id: uuid.UUID, *, limit: int | None = None) -> list[Note]:
        rows = self._sql.query(
            self._ordered(parent_id, limit), SQLResultFormats.all_scalar_list
        )
        return [self._from_dto(row) for row in self._ascending(rows, limit)]

    async def _aget(
        self, parent_id: uuid.UUID, *, limit: int | None = None
    ) -> list[Note]:
        rows = await self._sql.aquery(
            self._ordered(parent_id, limit), SQLResultFormats.all_scalar_list
        )
        return [self._from_dto(row) for row in self._ascending(rows, limit)]

    def _count(self, parent_id: uuid.UUID) -> int:
        return self._sql.query(self._counting(parent_id), lambda r: r.scalar()) or 0

    async def _acount(self, parent_id: uuid.UUID) -> int:
        return (
            await self._sql.aquery(self._counting(parent_id), lambda r: r.scalar())
        ) or 0

    def _next_sequence(self, parent_id: uuid.UUID) -> int:
        highest = self._sql.query(self._highest(parent_id), lambda r: r.scalar())
        return (highest or 0) + 1

    async def _anext_sequence(self, parent_id: uuid.UUID) -> int:
        highest = await self._sql.aquery(self._highest(parent_id), lambda r: r.scalar())
        return (highest or 0) + 1

    def _clear(self, parent_id: uuid.UUID) -> None:
        self._sql.delete_many(
            delete(self._model).where(self._model.parent_id == parent_id)
        )

    async def _aclear(self, parent_id: uuid.UUID) -> None:
        await self._sql.adelete_many(
            delete(self._model).where(self._model.parent_id == parent_id)
        )

    def _replace_range(
        self,
        parent_id: uuid.UUID,
        *,
        up_to_sequence: int,
        replacements: list[Note],
    ) -> list[Note]:
        rows = self._sql.execute_in_transaction(
            statements=[self._range_delete(parent_id, up_to_sequence)],
            rows_to_add=[self._to_dto(parent_id, r) for r in replacements],
        )
        return [self._from_dto(row) for row in rows]

    async def _areplace_range(
        self,
        parent_id: uuid.UUID,
        *,
        up_to_sequence: int,
        replacements: list[Note],
    ) -> list[Note]:
        rows = await self._sql.aexecute_in_transaction(
            statements=[self._range_delete(parent_id, up_to_sequence)],
            rows_to_add=[self._to_dto(parent_id, r) for r in replacements],
        )
        return [self._from_dto(row) for row in rows]

    def _range_delete(self, parent_id: uuid.UUID, up_to_sequence: int) -> Any:
        return delete(self._model).where(
            self._model.parent_id == parent_id,
            self._model.sequence <= up_to_sequence,
        )


@pytest.fixture
def note_model(infra_settings: Any, unique: str) -> Iterator[type[Any]]:
    model = build_note_model(f"note_{unique}")
    service = SQLService()
    with service._get_session() as session:
        model.metadata.create_all(session.get_bind())
    yield model
    with service._get_session() as session:
        model.metadata.drop_all(session.get_bind())


@pytest.fixture
def notes(note_model: type[Any]) -> SqlNoteStore:
    return SqlNoteStore(note_model)


@pytest.fixture
def parent() -> uuid.UUID:
    return uuid.uuid4()


def seed(store: SqlNoteStore, parent: uuid.UUID, count: int) -> list[Note]:
    return store.append(
        parent, [Note(sequence=i, body=f"n{i}") for i in range(1, count + 1)]
    )


@pytest.mark.integration
def test_append_stores_every_item_in_order(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    seed(notes, parent, 3)
    assert [n.body for n in notes.get(parent)] == ["n1", "n2", "n3"]


@pytest.mark.integration
async def test_aappend_stores_every_item_in_order(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    await notes.aappend(parent, [Note(sequence=i, body=f"n{i}") for i in (1, 2)])
    assert [n.body for n in await notes.aget(parent)] == ["n1", "n2"]


@pytest.mark.integration
def test_a_limit_takes_the_most_recent_items_in_ascending_order(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    seed(notes, parent, 5)
    assert [n.sequence for n in notes.get(parent, limit=2)] == [4, 5]


@pytest.mark.integration
async def test_an_async_limit_takes_the_most_recent_items_in_ascending_order(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    seed(notes, parent, 5)
    assert [n.sequence for n in await notes.aget(parent, limit=3)] == [3, 4, 5]


@pytest.mark.integration
def test_a_parent_with_no_rows_reads_as_empty(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    assert notes.get(parent) == []
    assert notes.count(parent) == 0
    assert notes.next_sequence(parent) == 1


@pytest.mark.integration
def test_count_and_next_sequence_follow_what_was_stored(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    seed(notes, parent, 4)
    assert notes.count(parent) == 4
    assert notes.next_sequence(parent) == 5


@pytest.mark.integration
async def test_acount_and_anext_sequence_follow_what_was_stored(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    seed(notes, parent, 4)
    assert await notes.acount(parent) == 4
    assert await notes.anext_sequence(parent) == 5


@pytest.mark.integration
def test_one_parent_s_rows_never_reach_another(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    other = uuid.uuid4()
    seed(notes, parent, 3)
    seed(notes, other, 1)
    assert notes.count(parent) == 3
    assert notes.next_sequence(other) == 2


@pytest.mark.integration
def test_clear_removes_the_parent_s_rows_only(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    other = uuid.uuid4()
    seed(notes, parent, 3)
    seed(notes, other, 2)
    notes.clear(parent)
    assert notes.count(parent) == 0
    assert notes.count(other) == 2


@pytest.mark.integration
async def test_aclear_removes_the_parent_s_rows(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    seed(notes, parent, 2)
    await notes.aclear(parent)
    assert await notes.acount(parent) == 0


@pytest.mark.integration
def test_clearing_a_parent_with_nothing_stored_is_a_no_op(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    notes.clear(parent)
    assert notes.count(parent) == 0


@pytest.mark.integration
def test_replace_range_swaps_the_rows_up_to_the_boundary(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    seed(notes, parent, 4)
    notes.replace_range(
        parent, up_to_sequence=2, replacements=[Note(sequence=1, body="summary")]
    )
    assert [n.body for n in notes.get(parent)] == ["summary", "n3", "n4"]


@pytest.mark.integration
async def test_areplace_range_swaps_the_rows_up_to_the_boundary(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    seed(notes, parent, 4)
    await notes.areplace_range(
        parent, up_to_sequence=2, replacements=[Note(sequence=1, body="summary")]
    )
    assert [n.body for n in await notes.aget(parent)] == ["summary", "n3", "n4"]


@pytest.mark.integration
def test_replace_range_is_validated_before_anything_is_written(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    seed(notes, parent, 3)
    with pytest.raises(ValueError, match="use clear\\(\\)"):
        notes.replace_range(parent, up_to_sequence=2, replacements=[])
    with pytest.raises(ValueError, match="exceeds"):
        notes.replace_range(
            parent, up_to_sequence=2, replacements=[Note(sequence=9, body="x")]
        )
    assert notes.count(parent) == 3


@pytest.mark.integration
def test_a_replace_range_that_fails_leaves_the_rows_it_would_have_deleted(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    """Delete and insert share one transaction, so a refused insert keeps both."""
    seed(notes, parent, 3)
    with pytest.raises(SQLConflictError):
        notes.replace_range(
            parent,
            up_to_sequence=2,
            replacements=[Note(sequence=1, body="a"), Note(sequence=1, body="b")],
        )
    assert [n.body for n in notes.get(parent)] == ["n1", "n2", "n3"]


@pytest.mark.integration
async def test_two_concurrent_appends_both_land_when_their_sequences_differ(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    await asyncio.gather(
        *(notes.aappend(parent, [Note(sequence=i, body=f"n{i}")]) for i in range(1, 9))
    )
    assert await notes.acount(parent) == 8
    assert [n.sequence for n in await notes.aget(parent)] == list(range(1, 9))


@pytest.mark.integration
async def test_two_appends_racing_for_one_sequence_leave_exactly_one_row(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    """The unique index is what makes a reused sequence a failure, not a loss."""
    results = await asyncio.gather(
        notes.aappend(parent, [Note(sequence=1, body="first")]),
        notes.aappend(parent, [Note(sequence=1, body="second")]),
        return_exceptions=True,
    )
    failures = [r for r in results if isinstance(r, BaseException)]
    assert len(failures) == 1
    assert isinstance(failures[0], SQLConflictError)
    assert await notes.acount(parent) == 1


@pytest.mark.integration
async def test_sequences_taken_under_a_lock_are_never_reused(
    notes: SqlNoteStore, parent: uuid.UUID
) -> None:
    """`next_sequence` is only safe when callers serialise around it."""
    lock = asyncio.Lock()

    async def append_one(body: str) -> None:
        async with lock:
            sequence = await notes.anext_sequence(parent)
            await notes.aappend(parent, [Note(sequence=sequence, body=body)])

    await asyncio.gather(*(append_one(f"n{i}") for i in range(10)))
    assert [n.sequence for n in await notes.aget(parent)] == list(range(1, 11))
