"""`tools/misc.py`: the small coercions every service reaches for.

Most of these read values that arrived as strings — an environment variable, a
query parameter, a tool-call argument — so what they do with a blank, a `None`
and a malformed value is the whole of their behaviour.
"""

import asyncio
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID

import pytest

from base.tools.misc import (
    _make_json_serializable,
    asleep_until,
    chunk_list,
    decode_base64,
    generic_deep_deserialize,
    getint,
    human_readable_duration_from_minutes,
    is_json,
    isfloat,
    isint,
    normalize_emoji_colors,
    parse_arguments,
    parse_json_pairs,
    parse_json_text,
    parse_uuid,
    sort_by_order_key,
    split_csv,
    string_truncate,
    stringify_document_values,
    try_parse_json,
)


def test_split_csv_drops_blanks_and_surrounding_spacing() -> None:
    assert split_csv(" a , b ,, c , ") == ["a", "b", "c"]


def test_split_csv_lowercases_only_when_asked() -> None:
    assert split_csv("OpenAI, Azure") == ["OpenAI", "Azure"]
    assert split_csv("OpenAI, Azure", lower=True) == ["openai", "azure"]


def test_split_csv_reads_an_unset_setting_as_no_values() -> None:
    assert split_csv(None) == []
    assert split_csv("") == []
    assert split_csv("  ,  ") == []


def test_split_csv_keeps_a_single_value_without_a_comma() -> None:
    assert split_csv("solo") == ["solo"]


def test_parse_uuid_passes_a_uuid_straight_through() -> None:
    value = UUID(int=9)
    assert parse_uuid(value) is value


def test_parse_uuid_reads_a_well_formed_string() -> None:
    assert parse_uuid("00000000-0000-0000-0000-000000000009") == UUID(int=9)


@pytest.mark.parametrize("value", [None, "", "not-a-uuid", "123", 7])
def test_parse_uuid_answers_none_rather_than_raising(value: object) -> None:
    assert parse_uuid(value) is None  # type: ignore[arg-type]


def test_chunk_list_splits_a_list_into_runs_of_the_requested_size() -> None:
    assert list(chunk_list([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunk_list_yields_nothing_for_an_empty_list() -> None:
    assert list(chunk_list([], 3)) == []


def test_a_size_larger_than_the_list_yields_one_chunk() -> None:
    assert list(chunk_list([1, 2], 10)) == [[1, 2]]


def test_a_size_of_one_yields_every_item_on_its_own() -> None:
    assert list(chunk_list([1, 2, 3], 1)) == [[1], [2], [3]]


def test_a_size_of_zero_raises_rather_than_looping_forever() -> None:
    """Pinned as it stands: the error comes from `range`, not from a guard."""
    with pytest.raises(ValueError, match="must not be zero"):
        list(chunk_list([1, 2], 0))


def test_chunking_is_lazy_so_nothing_happens_until_it_is_iterated() -> None:
    generator = chunk_list([1, 2], 0)
    assert list(chunk_list([1], 1)) == [[1]]
    with pytest.raises(ValueError):
        next(generator)


@pytest.mark.parametrize("value", ["7", 7, 7.9, "-3"])
def test_isint_accepts_anything_int_can_read(value: object) -> None:
    assert isint(value) is True


@pytest.mark.parametrize("value", ["7.5", "x", None, [1]])
def test_isint_rejects_what_int_cannot_read(value: object) -> None:
    assert isint(value) is False


@pytest.mark.parametrize("value", ["7.5", 7, "1e3"])
def test_isfloat_accepts_anything_float_can_read(value: object) -> None:
    assert isfloat(value) is True


@pytest.mark.parametrize("value", ["x", None, {}])
def test_isfloat_rejects_what_float_cannot_read(value: object) -> None:
    assert isfloat(value) is False


def test_getint_reads_a_number_and_answers_none_for_anything_else() -> None:
    assert getint("12") == 12
    assert getint(None) is None
    assert getint("twelve") is None


def test_try_parse_json_reports_success_beside_the_value() -> None:
    assert try_parse_json('{"a": 1}') == (True, {"a": 1})
    assert try_parse_json("not json") == (False, None)
    assert try_parse_json(None) == (False, None)  # type: ignore[arg-type]


def test_is_json_answers_the_same_question_without_the_value() -> None:
    assert is_json("[1, 2]") is True
    assert is_json("{oops}") is False


def test_parse_json_pairs_decodes_what_it_can_and_keeps_the_rest_verbatim() -> None:
    assert parse_json_pairs(("a", "b", "c"), ['{"x": 1}', "plain", None]) == {
        "a": {"x": 1},
        "b": "plain",
        "c": None,
    }


def test_parse_arguments_reads_each_tool_argument_as_its_own_type() -> None:
    assert parse_arguments(
        {
            "quoted": '"7"',
            "number": "7",
            "decimal": "7.5",
            "flag": "TRUE",
            "payload": '{"a": 1}',
            "text": "cairo",
        }
    ) == {
        "quoted": "7",
        "number": 7,
        "decimal": 7.5,
        "flag": True,
        "payload": {"a": 1},
        "text": "cairo",
    }


def test_parse_json_text_returns_the_original_string_when_it_is_not_json() -> None:
    assert parse_json_text('{"a": 1}') == {"a": 1}
    assert parse_json_text("hello") == "hello"
    assert parse_json_text(None) is None


def test_deep_deserialize_unwraps_json_nested_inside_json() -> None:
    assert generic_deep_deserialize('{"outer": "{\\"inner\\": 1}"}') == {
        "outer": {"inner": 1}
    }


def test_deep_deserialize_leaves_a_plain_string_alone() -> None:
    assert generic_deep_deserialize("plain") == "plain"


def test_decode_base64_adds_the_padding_a_url_safe_token_dropped() -> None:
    assert decode_base64("aGVsbG8") == "hello"
    assert decode_base64("aGVsbG8=") == "hello"


def test_string_truncate_only_shortens_what_is_too_long() -> None:
    assert string_truncate("abcdef", 3) == "abc..."
    assert string_truncate("abc", 3) == "abc"
    assert string_truncate("abcdef", 3, tail="…") == "abc…"


def test_a_duration_reads_as_hours_and_minutes_with_matching_plurals() -> None:
    assert human_readable_duration_from_minutes(0) == "0 minutes"
    assert human_readable_duration_from_minutes(1) == "1 minute"
    assert human_readable_duration_from_minutes(61) == "1 hour and 1 minute"
    assert human_readable_duration_from_minutes(125) == "2 hours and 5 minutes"


def test_normalising_an_emoji_removes_its_skin_tone_modifier() -> None:
    assert normalize_emoji_colors("\U0001f44d\U0001f3fd") == "\U0001f44d"
    assert normalize_emoji_colors("\U0001f44d") == "\U0001f44d"


def test_stringify_document_values_leaves_strings_alone() -> None:
    assert stringify_document_values({"a": 1, "b": "two", "c": None}) == {
        "a": "1",
        "b": "two",
        "c": "None",
    }


def test_sorting_by_display_order_puts_entries_without_one_last() -> None:
    ordered = sort_by_order_key(
        {
            "third": {"display_order": 3},
            "first": {"display_order": 1},
            "unordered": "not a dict",
        }
    )
    assert list(ordered) == ["first", "third", "unordered"]


class Colour(Enum):
    RED = "red"


def test_making_a_value_json_serialisable_reaches_every_nesting() -> None:
    assert _make_json_serializable(
        {
            "when": datetime(2026, 1, 1, tzinfo=UTC),
            "who": UUID(int=2),
            "what": Colour.RED,
            "list": [UUID(int=3)],
            "tuple": (Colour.RED,),
            "set": {1},
            "bytes": b"hi",
        }
    ) == {
        "when": "2026-01-01T00:00:00+00:00",
        "who": "00000000-0000-0000-0000-000000000002",
        "what": "red",
        "list": ["00000000-0000-0000-0000-000000000003"],
        "tuple": ("red",),
        "set": [1],
        "bytes": "hi",
    }


def test_an_already_serialisable_value_is_returned_unchanged() -> None:
    assert _make_json_serializable(7) == 7
    assert _make_json_serializable("x") == "x"
    assert _make_json_serializable(None) is None


async def test_waiting_for_a_condition_reports_success_when_it_turns_true() -> None:
    state = {"ready": False}

    async def flip() -> None:
        await asyncio.sleep(0.01)
        state["ready"] = True

    task = asyncio.create_task(flip())
    assert await asleep_until(lambda: state["ready"], 1.0, 0.005) is True
    await task


async def test_waiting_gives_up_and_reports_failure_once_the_budget_is_spent() -> None:
    assert await asleep_until(lambda: False, 0.02, 0.005) is False


async def test_a_condition_already_true_is_not_waited_on_at_all() -> None:
    assert await asleep_until(lambda: True, 0.0, 10.0) is True
