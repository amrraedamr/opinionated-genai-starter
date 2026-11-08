"""`protected_fields`: the columns `edit` refuses to take from a caller.

Identity and the three timestamps belong to the store, not to whoever calls
it, and a caller that sends them — deliberately or because they round-tripped
a row they had read — must not be able to rewrite them. A subclass may protect
more, and that is the mechanism a service store uses to freeze a natural key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from base.services.store.sql import SQLStore

from .conftest import WidgetStore


@pytest.fixture
def existing(store: WidgetStore, widget: type[Any]) -> Any:
    return store.add(widget(name="alpha", note="original", size=1))


def hostile_patch(widget: type[Any], **values: Any) -> Any:
    """A patch that sets every protected column to something recognisable."""
    patch = widget(**values)
    patch.id = 424242
    patch.created_at = datetime(1999, 1, 1, tzinfo=UTC)
    patch.updated_at = datetime(1999, 1, 1, tzinfo=UTC)
    patch.deleted_at = datetime(1999, 1, 1, tzinfo=UTC)
    return patch


def test_the_default_protected_set_is_identity_timestamps_and_the_tenant() -> None:
    """`tenant_id` is here so an edit cannot move a row into another customer."""
    assert SQLStore.protected_fields == {
        "id",
        "created_at",
        "updated_at",
        "deleted_at",
        "tenant_id",
    }


def test_edit_keeps_the_stored_id(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    edited = store.edit(existing.id, hostile_patch(widget, name="alpha2"))
    assert edited is not None
    assert edited.id == existing.id
    assert store.get(existing.id).name == "alpha2"


def test_edit_keeps_the_stored_created_at(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    edited = store.edit(existing.id, hostile_patch(widget, name="alpha2"))
    assert edited is not None
    assert edited.created_at == existing.created_at


def test_edit_keeps_the_row_active_when_the_caller_sends_deleted_at(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    store.edit(existing.id, hostile_patch(widget, name="alpha2"))
    assert store.get(existing.id).deleted_at is None


def test_edit_refuses_the_updated_at_the_caller_sent(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    edited = store.edit(existing.id, hostile_patch(widget, name="alpha2"))
    assert edited is not None
    assert edited.updated_at.year != 1999


async def test_aedit_protects_the_same_columns(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    edited = await store.aedit(existing.id, hostile_patch(widget, name="alpha2"))
    assert edited is not None
    assert edited.id == existing.id
    assert edited.created_at == existing.created_at
    assert edited.deleted_at is None


def test_a_subclass_can_freeze_a_column_of_its_own(
    widget: type[Any], existing: Any
) -> None:
    class FrozenNameStore(WidgetStore):
        protected_fields: ClassVar[frozenset[str]] = SQLStore.protected_fields | {
            "name"
        }

    edited = FrozenNameStore(widget).edit(
        existing.id, widget(name="renamed", note="changed")
    )
    assert edited is not None
    assert edited.name == "alpha"
    assert edited.note == "changed"


def test_sanitize_drops_protected_and_unmapped_keys(store: WidgetStore) -> None:
    sanitized = store._sanitize_update_data(
        {
            "id": 1,
            "created_at": "x",
            "updated_at": "x",
            "deleted_at": "x",
            "name": "kept",
            "not_a_column": "dropped",
        }
    )
    assert sanitized == {"name": "kept"}


def test_the_store_knows_which_attributes_are_columns(store: WidgetStore) -> None:
    assert store._column_names == {
        "id",
        "name",
        "note",
        "size",
        "expires_at",
        "created_at",
        "updated_at",
        "deleted_at",
    }


def test_update_from_dict_ignores_a_key_the_model_does_not_have(
    existing: Any,
) -> None:
    existing.update_from_dict({"name": "renamed", "nonsense": 1})
    assert existing.name == "renamed"
    assert not hasattr(existing, "nonsense")
