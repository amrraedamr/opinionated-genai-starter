"""The shared `BaseModel`: snake_case in Python, camelCase on the wire.

Every view model in this repository inherits it, so what it does to names and
what it accepts on input is the contract every API response depends on.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from base.models.view_models import (
    BaseModel,
    BaseResponse,
    DebugContext,
    ErrorResponse,
    to_camel,
)


class Sample(BaseModel):
    user_id: int
    display_name: str = "anonymous"
    nested_items: list[str] = []


def test_field_names_stay_snake_case_in_python() -> None:
    assert Sample(user_id=1).user_id == 1
    assert set(Sample.model_fields) == {"user_id", "display_name", "nested_items"}


def test_dumping_by_alias_camelises_every_field() -> None:
    assert Sample(user_id=1).model_dump(by_alias=True) == {
        "userId": 1,
        "displayName": "anonymous",
        "nestedItems": [],
    }


def test_dumping_without_an_alias_keeps_the_snake_case_names() -> None:
    assert set(Sample(user_id=1).model_dump()) == {
        "user_id",
        "display_name",
        "nested_items",
    }


def test_serialize_camelises_by_default() -> None:
    assert Sample(user_id=2).serialize() == {
        "userId": 2,
        "displayName": "anonymous",
        "nestedItems": [],
    }


def test_serialize_can_be_asked_for_snake_case() -> None:
    assert "user_id" in Sample(user_id=2).serialize(by_alias=False)


def test_serialize_can_drop_unset_and_none_fields() -> None:
    class Optional(BaseModel):
        kept: int = 1
        dropped: str | None = None

    assert Optional().serialize(exclude_none=True) == {"kept": 1}
    assert Optional(dropped="x").serialize(exclude_unset=True) == {"dropped": "x"}


def test_jsonify_emits_camel_case_json() -> None:
    assert Sample(user_id=3).jsonify() == (
        '{"userId":3,"displayName":"anonymous","nestedItems":[]}'
    )


def test_jsonify_can_be_asked_for_snake_case_json() -> None:
    assert '"user_id":3' in Sample(user_id=3).jsonify(by_alias=False)


def test_both_the_alias_and_the_field_name_are_accepted_on_input() -> None:
    assert Sample(userId=7).user_id == 7  # type: ignore[call-arg]
    assert Sample(user_id=7).user_id == 7


def test_a_name_that_is_neither_the_field_nor_its_alias_is_ignored() -> None:
    """`extra` is left at pydantic's default, so a misspelt key is dropped."""
    assert Sample(user_id=1, userid=2).user_id == 1  # type: ignore[call-arg]


def test_a_wrong_type_is_still_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(user_id="not a number")  # type: ignore[arg-type]


def test_deserialize_fills_defaults_and_skips_validation() -> None:
    built = Sample.deserialize({"user_id": "9"})
    assert built.user_id == "9"
    assert built.display_name == "anonymous"


def test_to_camel_leaves_a_single_word_alone() -> None:
    assert to_camel("name") == "name"
    assert to_camel("user_id") == "userId"


def test_models_can_be_built_from_arbitrary_objects_attributes() -> None:
    class Row:
        user_id = 11
        display_name = "row"
        nested_items: list[str] = []

    assert Sample.model_validate(Row()).user_id == 11


def test_the_error_envelope_reports_failure_and_camelises_its_debug_block() -> None:
    response: ErrorResponse[Any] = ErrorResponse(
        error="boom",
        debug=DebugContext(request_id="r1", path="/x", method="GET"),
    )
    dumped = response.serialize(exclude_none=True)
    assert dumped["success"] is False
    assert dumped["error"] == "boom"
    assert dumped["debug"]["requestId"] == "r1"


def test_the_success_envelope_defaults_to_success_with_no_data() -> None:
    response: BaseResponse[int] = BaseResponse()
    assert response.success is True
    assert response.data is None
