"""`serializers.py` and the decoder that is meant to be its inverse.

Caching, hashing and the broker all travel through `serialize`, so what it can
carry decides what can be cached. The round trip is checked type by type, and
the one type that does not survive it is pinned as it stands.
"""

import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from base.models.view_models import BaseModel
from base.tools.deserializers import deserialize
from base.tools.serializers import (
    ExpansiveEncoder,
    encode_base_model,
    encode_datetime,
    encode_type,
    encode_uuid,
    serialize,
)


class Payload(BaseModel):
    name: str
    count: int = 0


def test_a_plain_structure_serialises_as_ordinary_json() -> None:
    assert json.loads(serialize({"a": [1, True, None]})) == {"a": [1, True, None]}


def test_a_string_is_quoted_so_it_reads_back_as_a_string() -> None:
    assert serialize("already a string") == '"already a string"'
    assert deserialize(serialize("already a string")) == "already a string"


def test_a_string_holding_json_reads_back_as_that_string() -> None:
    assert deserialize(serialize('{"a": 1}')) == '{"a": 1}'


def test_a_datetime_is_tagged_so_it_can_be_read_back() -> None:
    assert json.loads(serialize(datetime(2026, 1, 1, tzinfo=UTC))) == {
        "_type": "datetime",
        "value": "2026-01-01T00:00:00+00:00",
    }


def test_a_datetime_survives_the_round_trip() -> None:
    moment = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    assert deserialize(serialize(moment)) == moment


def test_a_uuid_survives_the_round_trip() -> None:
    assert deserialize(serialize(UUID(int=12))) == UUID(int=12)


def test_a_type_survives_the_round_trip_as_the_same_class() -> None:
    assert deserialize(serialize(Payload)) is Payload


def test_tagged_values_nested_inside_a_structure_survive_the_round_trip() -> None:
    original = {"when": datetime(2026, 1, 1, tzinfo=UTC), "ids": [UUID(int=1)]}
    assert deserialize(serialize(original)) == original


def test_a_pydantic_model_is_tagged_with_its_class_and_its_values() -> None:
    encoded = json.loads(serialize(Payload(name="x", count=2)))
    assert encoded["_type"] == "base_model"
    assert encoded["_class"].endswith(".Payload")
    assert encoded["value"] == {"name": "x", "count": 2}


def test_a_pydantic_model_survives_the_round_trip() -> None:
    """The encoder tags `base_model`; the decoder has to know the tag.

    Anything that stores a model through this pair — the cache above all —
    read back a tagged dict rather than the model it wrote.
    """
    restored = deserialize(serialize(Payload(name="x", count=2)))
    assert isinstance(restored, Payload)
    assert restored.name == "x"
    assert restored.count == 2


def test_a_nested_model_survives_too() -> None:
    """A model inside a container is decoded on the way past."""
    payload = {"items": [Payload(name="a", count=1)], "when": datetime(2030, 1, 1)}
    restored = deserialize(serialize(payload))

    assert isinstance(restored["items"][0], Payload)
    assert restored["items"][0].name == "a"
    assert restored["when"] == datetime(2030, 1, 1)


def test_a_type_this_encoder_does_not_know_is_refused() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        serialize(object())


def test_a_set_is_refused_rather_than_silently_ordered() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        serialize({"tags": {"a", "b"}})


def test_serialize_passes_json_dumps_options_through() -> None:
    assert serialize({"b": 1, "a": 2}, sort_keys=True) == '{"a": 2, "b": 1}'


def test_the_encoder_registers_the_four_types_core_always_carries() -> None:
    from pydantic import BaseModel as PydanticBaseModel

    assert {datetime, PydanticBaseModel, UUID, type} <= set(ExpansiveEncoder.encoders)


def test_the_encoder_registers_the_optional_extras_this_build_installs() -> None:
    from fastapi import UploadFile
    from sqlalchemy.orm import DeclarativeBase

    assert UploadFile in ExpansiveEncoder.encoders
    assert DeclarativeBase in ExpansiveEncoder.encoders


def test_each_encoder_writes_the_tag_its_decoder_looks_for() -> None:
    assert encode_datetime(datetime(2026, 1, 1, tzinfo=UTC))["_type"] == "datetime"
    assert encode_uuid(UUID(int=1))["_type"] == "uuid"
    assert encode_type(Payload)["_type"] == "type"
    assert encode_base_model(Payload(name="x"))["_type"] == "base_model"


def test_a_short_numeric_string_is_read_back_as_a_number() -> None:
    """`deserialize` short-circuits anything under ten characters that is an int."""
    assert deserialize("42") == 42


def test_a_short_non_numeric_string_falls_through_to_json() -> None:
    assert deserialize('"hi"') == "hi"


def test_a_subclass_of_a_registered_type_uses_that_types_encoder() -> None:
    class Stamped(datetime):
        pass

    encoded = json.loads(serialize(Stamped(2026, 1, 1, tzinfo=UTC)))
    assert encoded["_type"] == "datetime"
