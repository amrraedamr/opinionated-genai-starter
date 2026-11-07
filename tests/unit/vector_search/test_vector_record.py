"""The record every vector backend speaks, and the names it carries.

The field is `metadata`, not `payload`: `payload` is Qdrant's own word and
naming it here would have leaked one backend's vocabulary into the shared
model. Each adapter translates on the way down. The id is a string, which is
what every backend can represent, and these tests pin that the names and the
round trip do not drift.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from base.models.view_models.vector_search import VectorRecord, VectorSearchResult


def test_a_record_carries_id_vector_and_metadata() -> None:
    assert set(VectorRecord.model_fields) == {"id", "vector", "metadata"}


def test_a_record_has_no_payload_field() -> None:
    """`payload` is Qdrant's word; the shared model does not borrow it."""
    assert "payload" not in VectorRecord.model_fields


def test_metadata_defaults_to_an_empty_mapping() -> None:
    assert VectorRecord(id="a", vector=[0.1]).metadata == {}


def test_each_record_gets_its_own_default_metadata() -> None:
    first = VectorRecord(id="a", vector=[0.1])
    first.metadata["x"] = 1
    assert VectorRecord(id="b", vector=[0.1]).metadata == {}


def test_metadata_round_trips_through_a_dump_and_back() -> None:
    record = VectorRecord(
        id="a", vector=[0.1, 0.2], metadata={"text": "héllo 🎉", "n": 3, "ok": True}
    )
    assert VectorRecord(**record.model_dump()) == record


def test_nested_metadata_survives_the_round_trip() -> None:
    metadata: dict[str, Any] = {"outer": {"inner": [1, {"deep": None}]}}
    record = VectorRecord(id="a", vector=[0.1], metadata=metadata)
    assert VectorRecord(**record.model_dump()).metadata == metadata


def test_an_id_must_be_a_string() -> None:
    """Pins the shape: an unsigned integer point id cannot be expressed here."""
    with pytest.raises(ValidationError):
        VectorRecord(id=7, vector=[0.1])  # type: ignore[arg-type]


def test_a_vector_is_required() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(id="a")  # type: ignore[call-arg]


def test_integers_in_a_vector_are_read_as_floats() -> None:
    assert VectorRecord(id="a", vector=[1, 2]).vector == [1.0, 2.0]


def test_a_result_adds_a_score_to_the_record_shape() -> None:
    assert set(VectorSearchResult.model_fields) == {"id", "score", "vector", "metadata"}


def test_a_result_may_omit_the_vector() -> None:
    assert VectorSearchResult(id="a", score=0.5).vector is None


def test_a_result_round_trips_through_a_dump_and_back() -> None:
    result = VectorSearchResult(
        id="a", score=0.75, vector=[0.1], metadata={"text": "hi"}
    )
    assert VectorSearchResult(**result.model_dump()) == result


def test_a_score_is_required() -> None:
    with pytest.raises(ValidationError):
        VectorSearchResult(id="a")  # type: ignore[call-arg]
