"""`hash_args`: the key `KeyedSingleton` derives when a caller names none.

Two callers asking for the same thing must land on the same instance, so the
hash has to ignore the order a dict was built in and has to see through the
types JSON cannot carry on its own.
"""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from base.models.view_models import BaseModel
from base.tools.hashing import hash_args


class Config(BaseModel):
    model_name: str
    temperature: float = 0.0


def test_the_same_arguments_always_hash_the_same_way() -> None:
    assert hash_args("a", 1, flag=True) == hash_args("a", 1, flag=True)


def test_the_hash_is_a_sha256_digest() -> None:
    digest = hash_args("a")
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_different_arguments_hash_differently() -> None:
    assert hash_args("a") != hash_args("b")
    assert hash_args(1) != hash_args("1")
    assert hash_args(a=1) != hash_args(b=1)


def test_keyword_order_does_not_change_the_hash() -> None:
    assert hash_args(alpha=1, beta=2) == hash_args(beta=2, alpha=1)


def test_key_order_inside_a_dict_argument_does_not_change_the_hash() -> None:
    assert hash_args({"alpha": 1, "beta": 2}) == hash_args({"beta": 2, "alpha": 1})


def test_key_order_inside_a_nested_dict_does_not_change_the_hash() -> None:
    assert hash_args(cfg={"outer": {"a": 1, "b": 2}}) == hash_args(
        cfg={"outer": {"b": 2, "a": 1}}
    )


def test_positional_order_does_change_the_hash() -> None:
    assert hash_args("a", "b") != hash_args("b", "a")


def test_an_argument_passed_by_name_hashes_differently_than_by_position() -> None:
    """Args and kwargs are hashed in separate slots, so the two are not equal."""
    assert hash_args("gpt-5") != hash_args(model="gpt-5")


def test_a_pydantic_model_hashes_by_its_values() -> None:
    assert hash_args(Config(model_name="gpt-5")) == hash_args(
        Config(model_name="gpt-5")
    )
    assert hash_args(Config(model_name="gpt-5")) != hash_args(
        Config(model_name="gpt-4o")
    )


def test_a_model_differing_only_in_a_defaulted_field_hashes_differently() -> None:
    assert hash_args(Config(model_name="gpt-5")) != hash_args(
        Config(model_name="gpt-5", temperature=0.5)
    )


def test_a_uuid_hashes_by_its_value_not_its_identity() -> None:
    assert hash_args(UUID(int=1)) == hash_args(UUID(int=1))
    assert hash_args(UUID(int=1)) != hash_args(UUID(int=2))


def test_a_datetime_hashes_by_its_instant() -> None:
    assert hash_args(datetime(2026, 1, 1, tzinfo=UTC)) == hash_args(
        datetime(2026, 1, 1, tzinfo=UTC)
    )
    assert hash_args(datetime(2026, 1, 1, tzinfo=UTC)) != hash_args(
        datetime(2026, 1, 2, tzinfo=UTC)
    )


def test_a_type_hashes_by_its_import_path() -> None:
    assert hash_args(Config) == hash_args(Config)
    assert hash_args(Config) != hash_args(BaseModel)


def test_no_arguments_at_all_still_produce_a_key() -> None:
    assert hash_args() == hash_args()
    assert hash_args() != hash_args(None)


def test_a_value_json_cannot_carry_is_refused_rather_than_hashed_loosely() -> None:
    """A silently-equal hash for two different objects would collide clients."""
    with pytest.raises(TypeError, match="not JSON serializable"):
        hash_args(object())


def test_the_hash_is_stable_across_processes() -> None:
    """A singleton key computed in one worker must match the next one's."""
    import subprocess
    import sys

    script = (
        "from base.tools.hashing import hash_args;"
        "print(hash_args('a', 1, flag=True, cfg={'b': 2, 'a': 1}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == hash_args("a", 1, flag=True, cfg={"b": 2, "a": 1})
