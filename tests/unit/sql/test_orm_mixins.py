"""The two ORM mixins every declarative root in this workspace carries.

`SerializableMixin` is what `SQLStore.edit` round-trips a patch through, so
what it does to a value on the way out decides what `edit` can write.
`TimestampMixin` owns the three columns the store protects and the soft-delete
flag every read filters on.
"""

from __future__ import annotations

import enum
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import DateTime, Integer, String, Uuid
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    mapped_column,
)

from base.models.orm import SerializableMixin, TimestampMixin
from base.services.sql.service import SQLService


class Flavour(enum.StrEnum):
    SWEET = "sweet"
    SOUR = "sour"


@pytest.fixture
def plain_model() -> type[Any]:
    """A model with one column of every kind `to_dict` has to convert."""

    class Base(MappedAsDataclass, DeclarativeBase, SerializableMixin):
        pass

    class Sample(TimestampMixin, Base):
        __tablename__ = "orm_mixin_sample"

        id: Mapped[int] = mapped_column(
            Integer, primary_key=True, autoincrement=True, init=False
        )
        name: Mapped[str] = mapped_column(String(50))
        ref: Mapped[uuid.UUID | None] = mapped_column(Uuid, default=None)
        flavour: Mapped[Flavour] = mapped_column(String(10), default=Flavour.SWEET)
        at: Mapped[datetime | None] = mapped_column(
            DateTime(timezone=True), default=None
        )

    return Sample


def test_to_dict_carries_every_mapped_column(plain_model: type[Any]) -> None:
    assert set(plain_model(name="a").to_dict()) == {
        "id",
        "name",
        "ref",
        "flavour",
        "at",
        "created_at",
        "updated_at",
        "deleted_at",
    }


def test_to_dict_drops_what_it_is_asked_to_exclude(plain_model: type[Any]) -> None:
    data = plain_model(name="a").to_dict(exclude={"id", "created_at"})
    assert "id" not in data
    assert "created_at" not in data
    assert data["name"] == "a"


def test_to_dict_renders_a_datetime_as_an_iso_string(plain_model: type[Any]) -> None:
    moment = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert plain_model(name="a", at=moment).to_dict()["at"] == moment.isoformat()


def test_to_dict_renders_a_uuid_as_a_string(plain_model: type[Any]) -> None:
    ref = uuid.uuid4()
    assert plain_model(name="a", ref=ref).to_dict()["ref"] == str(ref)


def test_to_dict_renders_an_enum_as_its_value(plain_model: type[Any]) -> None:
    assert plain_model(name="a", flavour=Flavour.SOUR).to_dict()["flavour"] == "sour"


def test_to_dict_leaves_none_alone(plain_model: type[Any]) -> None:
    assert plain_model(name="a").to_dict()["ref"] is None


def test_update_from_dict_writes_the_keys_the_model_has(
    plain_model: type[Any],
) -> None:
    row = plain_model(name="a")
    row.update_from_dict({"name": "b", "flavour": Flavour.SOUR})
    assert (row.name, row.flavour) == ("b", Flavour.SOUR)


def test_update_from_dict_skips_a_key_the_model_does_not_have(
    plain_model: type[Any],
) -> None:
    row = plain_model(name="a")
    row.update_from_dict({"nonsense": 1})
    assert not hasattr(row, "nonsense")


def test_update_from_dict_of_nothing_changes_nothing(plain_model: type[Any]) -> None:
    row = plain_model(name="a")
    row.update_from_dict({})
    assert row.name == "a"


def test_clone_refuses_a_timestamped_model_with_its_default_exclusions(
    plain_model: type[Any],
) -> None:
    """A defect, pinned as it stands.

    `clone` excludes only `id`, so the three timestamp columns stay in the
    dict it passes to `__init__`. They are declared `init=False`, so no model
    carrying `TimestampMixin` — which is every model in this workspace — can
    be cloned without naming all four exclusions by hand.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument 'created_at'"):
        plain_model(name="a").clone()


def test_clone_copies_a_row_when_every_generated_column_is_excluded(
    plain_model: type[Any],
) -> None:
    original = plain_model(name="a", flavour=Flavour.SOUR)
    copy = original.clone(exclude={"id", "created_at", "updated_at", "deleted_at"})
    assert copy is not original
    assert (copy.name, copy.flavour) == ("a", Flavour.SOUR)


def test_soft_delete_stamps_a_timezone_aware_utc_moment(
    plain_model: type[Any],
) -> None:
    row = plain_model(name="a")
    assert row.deleted_at is None
    row.soft_delete()
    assert row.deleted_at is not None
    assert row.deleted_at.tzinfo is not None
    assert row.deleted_at.utcoffset() == UTC.utcoffset(None)


def test_restore_clears_the_stamp(plain_model: type[Any]) -> None:
    row = plain_model(name="a")
    row.soft_delete()
    row.restore()
    assert row.deleted_at is None


def test_restoring_a_row_that_was_never_deleted_is_a_no_op(
    plain_model: type[Any],
) -> None:
    row = plain_model(name="a")
    row.restore()
    assert row.deleted_at is None


def test_the_timestamps_are_unset_until_the_row_is_written(
    plain_model: type[Any],
) -> None:
    row = plain_model(name="a")
    assert row.created_at is None
    assert row.updated_at is None


def test_insert_sets_both_timestamps(widget: type[Any]) -> None:
    row = SQLService().upsert_row(widget(name="a"))
    assert isinstance(row.created_at, datetime)
    assert isinstance(row.updated_at, datetime)


def test_an_update_moves_updated_at_and_leaves_created_at(widget: type[Any]) -> None:
    service = SQLService()
    row = service.upsert_row(widget(name="a"))
    created, first_update = row.created_at, row.updated_at

    time.sleep(0.01)
    row.note = "changed"
    updated = service.upsert_row(row)

    assert updated.created_at == created
    assert updated.updated_at > first_update
