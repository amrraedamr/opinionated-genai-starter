"""`SQLResultFormats`: the shapes a query result is read into.

Every store method passes one of these to `SQLService.query`, so which one is
chosen decides whether a caller gets entities, rows or mappings. They are
exercised through a real result rather than a stub, because what they wrap is
SQLAlchemy's own `Result` behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.engine.row import Row, RowMapping
from sqlalchemy.exc import MultipleResultsFound

from base.models.view_models.exceptions.sql import SQLUnexpectedError
from base.services.sql.service import SQLService
from base.tools.database import SQLResultFormats


@pytest.fixture
def seeded(widget: type[Any]) -> type[Any]:
    SQLService().upsert_rows([widget(name=f"w{i}", size=i) for i in range(3)])
    return widget


def test_one_or_none_returns_the_first_row(seeded: type[Any]) -> None:
    row = SQLService().query(
        select(seeded.name, seeded.size).order_by(seeded.size),
        SQLResultFormats.one_or_none,
    )
    assert isinstance(row, Row)
    assert tuple(row) == ("w0", 0)


def test_one_or_none_is_none_when_nothing_matched(seeded: type[Any]) -> None:
    assert (
        SQLService().query(
            select(seeded).where(seeded.name == "absent"),
            SQLResultFormats.one_or_none,
        )
        is None
    )


def test_one_or_none_scalar_returns_the_entity(seeded: type[Any]) -> None:
    row = SQLService().query(
        select(seeded).where(seeded.name == "w1"), SQLResultFormats.one_or_none_scalar
    )
    assert row is not None
    assert row.name == "w1"


def test_one_or_none_scalar_is_none_when_nothing_matched(seeded: type[Any]) -> None:
    assert (
        SQLService().query(
            select(seeded).where(seeded.name == "absent"),
            SQLResultFormats.one_or_none_scalar,
        )
        is None
    )


def test_one_or_none_scalar_refuses_more_than_one_match(seeded: type[Any]) -> None:
    """A store reading by a column it believes unique finds out here if it is not."""
    with pytest.raises(SQLUnexpectedError) as caught:
        SQLService().query(select(seeded), SQLResultFormats.one_or_none_scalar)
    assert isinstance(caught.value.provider_error, MultipleResultsFound)


def test_all_scalar_list_returns_entities_in_a_list(seeded: type[Any]) -> None:
    rows = SQLService().query(
        select(seeded).order_by(seeded.size), SQLResultFormats.all_scalar_list
    )
    assert isinstance(rows, list)
    assert [row.name for row in rows] == ["w0", "w1", "w2"]


def test_all_scalar_sequence_returns_the_same_entities(seeded: type[Any]) -> None:
    rows = SQLService().query(
        select(seeded).order_by(seeded.size), SQLResultFormats.all_scalar_sequence
    )
    assert [row.name for row in rows] == ["w0", "w1", "w2"]


def test_all_returns_whole_rows(seeded: type[Any]) -> None:
    rows = SQLService().query(
        select(seeded.name, seeded.size).order_by(seeded.size), SQLResultFormats.all
    )
    assert [tuple(row) for row in rows] == [("w0", 0), ("w1", 1), ("w2", 2)]


def test_all_mappings_returns_rows_keyed_by_column(seeded: type[Any]) -> None:
    rows = SQLService().query(
        select(seeded.name, seeded.size).order_by(seeded.size),
        SQLResultFormats.all_mappings,
    )
    assert isinstance(rows[0], RowMapping)
    assert rows[0]["name"] == "w0"
    assert rows[0]["size"] == 0


def test_every_list_format_is_empty_when_nothing_matched(seeded: type[Any]) -> None:
    empty = select(seeded).where(seeded.name == "absent")
    service = SQLService()
    assert service.query(empty, SQLResultFormats.all_scalar_list) == []
    assert list(service.query(empty, SQLResultFormats.all_scalar_sequence)) == []
    assert service.query(empty, SQLResultFormats.all) == []
    assert service.query(empty, SQLResultFormats.all_mappings) == []
