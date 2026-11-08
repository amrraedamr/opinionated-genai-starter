"""The reusable ORM mixins in `models/orm`.

`SerializableMixin` is how a declarative row becomes something JSON can carry,
and `TimestampMixin` is where soft deletion lives, so both are exercised
against a real declarative model rather than a stand-in.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from base.models.orm import SerializableMixin, TimestampMixin


@dataclass
class Row(SerializableMixin):
    """A dataclass row, which is what `MappedAsDataclass` produces."""

    id: UUID = field(default_factory=uuid4)
    name: str = "widget"
    created_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_a_row_becomes_a_dict_json_can_carry() -> None:
    row = Row(id=UUID(int=3), name="widget")
    assert row.to_dict() == {
        "id": "00000000-0000-0000-0000-000000000003",
        "name": "widget",
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def test_named_columns_are_left_out_of_the_dict() -> None:
    assert "id" not in Row().to_dict(exclude={"id"})


def test_updating_from_a_dict_sets_only_attributes_that_exist() -> None:
    row = Row(name="old")
    row.update_from_dict({"name": "new", "not_a_column": "ignored"})
    assert row.name == "new"
    assert not hasattr(row, "not_a_column")


def test_cloning_drops_the_identifier_so_the_copy_is_a_new_row() -> None:
    original = Row(id=UUID(int=4), name="widget")
    clone = original.clone()
    assert clone.name == "widget"
    assert clone.id != original.id


def test_cloning_can_be_told_which_columns_to_drop() -> None:
    original = Row(id=UUID(int=5), name="dropped")
    clone = original.clone(exclude={"name"})
    assert clone.name == "widget"


def test_naming_another_column_does_not_keep_the_identifier() -> None:
    original = Row(id=UUID(int=8), name="widget")
    assert original.clone(exclude={"name"}).id != UUID(int=8)


def test_cloning_hands_back_the_column_types_rather_than_json_types() -> None:
    clone = Row(id=UUID(int=6), name="widget").clone(exclude={"name"})
    assert clone.created_at == datetime(2026, 1, 1, tzinfo=UTC)


def test_an_empty_exclude_set_still_drops_the_identifier() -> None:
    clone = Row(id=UUID(int=7), name="widget").clone(exclude=set())
    assert clone.id != UUID(int=7)


def test_the_timestamp_mixin_declares_the_three_lifecycle_columns() -> None:
    assert set(TimestampMixin.__annotations__) == {
        "created_at",
        "updated_at",
        "deleted_at",
    }


def test_soft_deleting_stamps_a_time_and_restoring_clears_it() -> None:
    class Record(TimestampMixin):
        def __init__(self) -> None:
            self.deleted_at = None

    record = Record()
    record.soft_delete()
    assert record.deleted_at is not None
    assert record.deleted_at.tzinfo is not None
    record.restore()
    assert record.deleted_at is None
