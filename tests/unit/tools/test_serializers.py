"""`serializers.py` and the decoder that is meant to be its inverse.

Caching, hashing and the broker all travel through `serialize`, so what it can
carry decides what can be cached. The round trip is checked type by type, and
the one type that does not survive it is pinned as it stands.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from base.models.view_models import BaseModel
from base.tools.deserializers import (
    ExpansiveDecoder,
    _import_cache,
    _init_params_cache,
    deserialize,
)
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


class Base(DeclarativeBase):
    pass


class Row(Base):
    """A declarative model, which the encoder tags `dto`."""

    __tablename__ = "serializer_round_trip"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(default="")


class Legacy:
    """A class whose constructor refuses a field its instances still carry."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0


class Sealed:
    """The same, with no room for a field the constructor did not take."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class Opaque:
    """Stands in for a class whose constructor cannot be inspected."""

    def __init__(self) -> None:
        self.name = ""


def path_of(cls: type) -> str:
    """The `_class` string the encoder writes for a class."""
    return f"{cls.__module__}.{cls.__qualname__}"


def tagged(tag: str, value: Any = None, **extra: Any) -> str:
    """One encoder envelope as JSON, including tags nothing here writes."""
    return json.dumps({"_type": tag, "value": value, **extra})


@pytest.fixture(autouse=True)
def _clear_decoder_caches() -> Iterator[None]:
    """The decoder caches resolved classes and signatures by path, process-wide."""
    _import_cache.clear()
    _init_params_cache.clear()
    yield
    _import_cache.clear()
    _init_params_cache.clear()


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


def test_a_number_is_read_back_as_a_number() -> None:
    assert deserialize("42") == 42
    assert deserialize(serialize(10**12)) == 10**12


def test_a_quoted_string_is_read_back_as_a_string() -> None:
    assert deserialize('"hi"') == "hi"


def test_a_string_of_digits_does_not_become_a_number() -> None:
    """`serialize` quotes it, so only the quoting decides the type."""
    assert deserialize(serialize("42")) == "42"


def test_input_that_is_not_json_is_refused_rather_than_guessed() -> None:
    with pytest.raises(json.JSONDecodeError):
        deserialize("007")


def test_a_subclass_of_a_registered_type_uses_that_types_encoder() -> None:
    class Stamped(datetime):
        pass

    encoded = json.loads(serialize(Stamped(2026, 1, 1, tzinfo=UTC)))
    assert encoded["_type"] == "datetime"


def test_a_declarative_row_survives_the_round_trip() -> None:
    restored = deserialize(serialize(Row(id=4, label="ingested")))

    assert isinstance(restored, Row)
    assert (restored.id, restored.label) == (4, "ingested")


def test_a_declarative_row_is_tagged_with_its_class_and_its_columns() -> None:
    encoded = json.loads(serialize(Row(id=1, label="a")))

    assert encoded["_type"] == "dto"
    assert encoded["_class"] == path_of(Row)
    assert encoded["value"] == {"id": 1, "label": "a"}


def test_the_app_model_tag_is_rebuilt_the_same_way_a_dto_is() -> None:
    """Nothing in this build writes `app_model`; a stored value still may."""
    restored = deserialize(
        tagged("app_model", {"name": "x", "count": 2}, _class=path_of(Payload))
    )

    assert isinstance(restored, Payload)
    assert restored.count == 2


def test_a_tagged_value_nested_in_a_row_is_decoded_on_the_way_past() -> None:
    envelope = tagged(
        "dto",
        {"id": 1, "label": {"_type": "uuid", "value": str(UUID(int=5))}},
        _class=path_of(Row),
    )

    assert deserialize(envelope).label == UUID(int=5)


def test_a_class_that_refuses_a_field_is_built_and_then_populated() -> None:
    restored = deserialize(
        tagged("dto", {"name": "a", "count": 3}, _class=path_of(Legacy))
    )

    assert (restored.name, restored.count) == ("a", 3)


def test_a_field_the_class_has_no_room_for_is_dropped_rather_than_forced() -> None:
    restored = deserialize(
        tagged("dto", {"name": "a", "unknown": 3}, _class=path_of(Sealed))
    )

    assert restored.name == "a"
    assert not hasattr(restored, "unknown")


def test_a_constructor_that_cannot_be_inspected_is_read_as_taking_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    def refuse(_: Any) -> None:
        raise ValueError("no signature found")

    monkeypatch.setattr(inspect, "signature", refuse)

    restored = deserialize(tagged("dto", {"name": "a"}, _class=path_of(Opaque)))

    assert restored.name == "a"


def test_a_class_is_resolved_once_and_then_read_from_the_cache() -> None:
    first = deserialize(tagged("type", path_of(Payload)))
    second = deserialize(tagged("type", path_of(Payload)))

    assert first is second is Payload
    assert _import_cache[path_of(Payload)] is Payload


def test_an_unknown_tag_is_left_as_the_dictionary_it_arrived_as() -> None:
    assert deserialize(tagged("mystery", {"a": 1})) == {
        "_type": "mystery",
        "value": {"a": 1},
    }


@pytest.mark.parametrize("value", [None, "", 0, [], {}])
def test_a_tag_carrying_no_value_decodes_to_nothing(value: Any) -> None:
    assert deserialize(tagged("uuid", value)) is None


def test_a_model_with_no_fields_decodes_to_nothing_rather_than_to_itself() -> None:
    """Pinned as it stands: an empty `value` is indistinguishable from absent."""

    class Empty(BaseModel):
        pass

    assert deserialize(tagged("base_model", {}, _class=path_of(Empty))) is None


def test_an_envelope_with_no_value_key_at_all_decodes_to_nothing() -> None:
    assert deserialize(json.dumps({"_type": "uuid"})) is None


def test_an_upload_is_tagged_by_name_rather_than_by_its_bytes() -> None:
    import io

    from fastapi import UploadFile

    encoded = json.loads(
        serialize(UploadFile(file=io.BytesIO(b"pdf"), filename="a.pdf"))
    )

    assert encoded == {
        "_type": "UploadFile",
        "filename": "a.pdf",
        "content_type": None,
    }


def test_an_upload_file_does_not_survive_the_round_trip() -> None:
    """The encoder writes `filename`, not `value`, and there is no decoder."""
    assert (
        deserialize(
            json.dumps(
                {"_type": "UploadFile", "filename": "a.pdf", "content_type": "x"}
            )
        )
        is None
    )


@pytest.mark.parametrize("value", ["not a date", "2026-13-45", "2026-01-01T99"])
def test_a_malformed_datetime_is_refused_rather_than_guessed(value: str) -> None:
    with pytest.raises(ValueError):
        deserialize(tagged("datetime", value))


def test_a_malformed_uuid_is_refused() -> None:
    with pytest.raises(ValueError):
        deserialize(tagged("uuid", "not-a-uuid"))


def test_a_type_tag_naming_a_module_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ModuleNotFoundError):
        deserialize(tagged("type", "no_such_module.Thing"))


def test_a_type_tag_naming_a_missing_attribute_is_refused() -> None:
    with pytest.raises(AttributeError):
        deserialize(tagged("type", f"{__name__}.NotDefinedHere"))


def test_a_type_tag_with_no_module_half_is_refused() -> None:
    with pytest.raises(ValueError):
        deserialize(tagged("type", "Payload"))


def test_a_model_envelope_without_its_class_is_refused() -> None:
    with pytest.raises(TypeError, match="_class"):
        deserialize(tagged("base_model", {"name": "x"}))


def test_an_extra_key_beside_a_tag_reaches_its_decoder_and_is_refused() -> None:
    """Extras are forwarded as keyword arguments, so an unknown one fails loudly."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        deserialize(tagged("uuid", str(UUID(int=1)), unexpected=1))


def test_a_dictionary_whose_type_is_null_is_read_as_an_ordinary_dictionary() -> None:
    assert deserialize(json.dumps({"_type": None, "value": {"a": 1}})) == {
        "_type": None,
        "value": {"a": 1},
    }


def test_a_key_named_type_in_ordinary_data_is_not_mistaken_for_a_tag() -> None:
    assert deserialize(json.dumps({"type": "datetime", "value": "nonsense"})) == {
        "type": "datetime",
        "value": "nonsense",
    }


def test_tags_nested_several_levels_deep_are_all_decoded() -> None:
    original = {
        "outer": {
            "models": [Payload(name="a"), Payload(name="b", count=2)],
            "when": {"start": datetime(2026, 1, 1, tzinfo=UTC)},
        },
        "ids": [[UUID(int=1)], [UUID(int=2)]],
    }

    restored = deserialize(serialize(original))

    assert [model.name for model in restored["outer"]["models"]] == ["a", "b"]
    assert restored["outer"]["when"]["start"] == datetime(2026, 1, 1, tzinfo=UTC)
    assert restored["ids"] == [[UUID(int=1)], [UUID(int=2)]]


def test_a_tuple_is_walked_the_same_way_a_list_is() -> None:
    """JSON never produces one, but the decoder is also called on live data."""
    decoded = ExpansiveDecoder.process_decoded_data(
        (1, {"_type": "uuid", "value": str(UUID(int=7))}, ({"a": 2},))
    )

    assert decoded == (1, UUID(int=7), ({"a": 2},))


def test_a_scalar_is_handed_back_untouched() -> None:
    assert ExpansiveDecoder.process_decoded_data("plain") == "plain"
    assert ExpansiveDecoder.process_decoded_data(7) == 7
    assert ExpansiveDecoder.process_decoded_data(None) is None


@pytest.mark.parametrize("value", [{}, [], ()])
def test_only_a_container_is_worth_walking(value: Any) -> None:
    assert ExpansiveDecoder.should_be_decoded(value) is True


@pytest.mark.parametrize("value", ["x", 1, 1.5, None, True])
def test_anything_that_is_not_a_container_is_left_alone(value: Any) -> None:
    assert ExpansiveDecoder.should_be_decoded(value) is False


def test_deserialize_passes_json_loads_options_through() -> None:
    assert deserialize("[1.5]", parse_float=str) == ["1.5"]


def test_an_empty_structure_survives_the_round_trip() -> None:
    assert deserialize(serialize({})) == {}
    assert deserialize(serialize([])) == []


def test_text_in_another_script_survives_the_round_trip() -> None:
    assert deserialize(serialize({"سؤال": "ما هي؟"})) == {"سؤال": "ما هي؟"}
