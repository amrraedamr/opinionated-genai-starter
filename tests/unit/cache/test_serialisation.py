"""What survives a round trip through the cache, and what does not.

Values go in through `serialize` and come back through `ExpansiveDecoder`, and
the two are not inverses. Each case here states which of the two it is, so a
change to either is visible rather than silent.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from base.models.view_models.exceptions.cache import CacheUnexpectedError
from base.services.cache.memory import InMemoryCacheService


class Payload(BaseModel):
    name: str
    count: int = 1


async def test_a_datetime_returns_as_a_datetime(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    moment = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    await memory_cache.aset("when", namespace, value=moment)
    assert await memory_cache.aget("when", namespace) == moment


async def test_a_uuid_returns_as_a_uuid(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    value = uuid4()
    await memory_cache.aset("id", namespace, value=value)
    assert await memory_cache.aget("id", namespace) == value


async def test_a_datetime_nested_in_a_dict_returns_as_a_datetime(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    moment = datetime(2026, 3, 4, tzinfo=UTC)
    await memory_cache.aset("nested", namespace, value={"at": moment, "n": 1})
    assert await memory_cache.aget("nested", namespace) == {"at": moment, "n": 1}


async def test_a_pydantic_model_comes_back_as_itself(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    """The encoder writes `base_model`, so the decoder has to read it.

    It used to register every other tag, and a model stored here came back
    as the envelope rather than the model.
    """
    await memory_cache.aset("model", namespace, value=Payload(name="x", count=2))

    restored = await memory_cache.aget("model", namespace)
    assert isinstance(restored, Payload)
    assert restored.name == "x"
    assert restored.count == 2


async def test_a_value_the_encoder_cannot_reach_is_refused_on_write(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    with pytest.raises(CacheUnexpectedError, match="not JSON serializable"):
        await memory_cache.aset("bad", namespace, value={1, 2, 3})
    assert await memory_cache.aget("bad", namespace) is None


async def test_an_arbitrary_object_is_refused_on_write(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    with pytest.raises(CacheUnexpectedError, match="not JSON serializable"):
        await memory_cache.aset("bad", namespace, value=object())


async def test_a_plain_string_reads_back_as_the_same_string(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset("greeting", namespace, value="hello")
    assert await memory_cache.aget("greeting", namespace) == "hello"


async def test_a_string_of_digits_stays_a_string(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset("numeric", namespace, value="123")
    assert await memory_cache.aget("numeric", namespace) == "123"


async def test_a_non_class_value_type_is_applied_to_the_stored_text(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    """The only way to read a stored string back unchanged."""
    await memory_cache.aset("greeting", namespace, value="hello")
    assert (
        await memory_cache.aget("greeting", namespace, value_type=str.upper) == "HELLO"
    )


async def test_a_class_value_type_is_ignored_in_favour_of_json_decoding(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    await memory_cache.aset("doc", namespace, value={"a": 1})
    assert await memory_cache.aget("doc", namespace, value_type=int) == {"a": 1}


async def test_unicode_and_emoji_survive_a_round_trip(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    value = {"greeting": "héllo wörld — 🎉🙂", "kanji": "日本語"}
    await memory_cache.aset("i18n", namespace, value=value)
    assert await memory_cache.aget("i18n", namespace) == value


async def test_a_large_value_survives_a_round_trip(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    value = {"blob": "x" * 500_000, "items": list(range(5_000))}
    await memory_cache.aset("big", namespace, value=value)
    assert await memory_cache.aget("big", namespace) == value


async def test_a_deeply_nested_structure_survives_a_round_trip(
    memory_cache: InMemoryCacheService, namespace: UUID
) -> None:
    value: dict = {"level": 0}
    for depth in range(1, 25):
        value = {"level": depth, "child": value}
    await memory_cache.aset("deep", namespace, value=value)
    assert await memory_cache.aget("deep", namespace) == value
