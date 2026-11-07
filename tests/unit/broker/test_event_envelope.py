"""The event envelope every service publishes and consumes.

`Event` crosses a process boundary as JSON, so what `Event.of` fills in, what
it refuses, and what survives `model_dump_json` are the whole of the contract
between a producer and a consumer that never share code.
"""

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from contracts.events import (
    CURRENT_EVENT_VERSION,
    SUPPORTED_EVENT_VERSIONS,
    Event,
    EventMeta,
)


class Body(BaseModel):
    name: str
    at: datetime


def test_of_fills_in_the_topic_and_payload() -> None:
    built = Event.of("user.registered", {"id": 7}, producer="identity")
    assert built.topic == "user.registered"
    assert built.payload == {"id": 7}


def test_of_records_the_producer() -> None:
    assert Event.of("t", {}, producer="identity").meta.producer == "identity"


def test_of_defaults_to_the_current_schema_version() -> None:
    assert Event.of("t", {}, producer="p").meta.schema_version == CURRENT_EVENT_VERSION


def test_of_defaults_to_no_correlation_id() -> None:
    assert Event.of("t", {}, producer="p").meta.correlation_id is None


def test_of_propagates_a_correlation_id() -> None:
    correlation = uuid4()
    built = Event.of("t", {}, producer="p", correlation_id=correlation)
    assert built.meta.correlation_id == correlation


def test_every_event_gets_its_own_id() -> None:
    ids = {Event.of("t", {}, producer="p").meta.event_id for _ in range(100)}
    assert len(ids) == 100


def test_occurred_at_is_timezone_aware() -> None:
    occurred = Event.of("t", {}, producer="p").meta.occurred_at
    assert occurred.tzinfo is not None
    assert occurred.utcoffset() == UTC.utcoffset(None)


def test_occurred_at_is_now() -> None:
    before = datetime.now(UTC)
    occurred = Event.of("t", {}, producer="p").meta.occurred_at
    assert before <= occurred <= datetime.now(UTC)


def test_a_model_payload_is_stored_in_json_mode() -> None:
    body = Body(name="ada", at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC))
    built = Event.of("t", body, producer="p")
    assert built.payload == {"name": "ada", "at": "2026-01-02T03:04:00Z"}


def test_a_dict_payload_is_stored_as_given() -> None:
    payload = {"nested": {"list": [1, 2, 3]}, "unicode": "é🎉"}
    assert Event.of("t", payload, producer="p").payload == payload


def test_an_empty_payload_is_allowed() -> None:
    assert Event.of("t", {}, producer="p").payload == {}


def test_a_payload_is_empty_by_default() -> None:
    assert Event(topic="t", meta=EventMeta(producer="p")).payload == {}


@pytest.mark.parametrize("version", [0, 2, 99, -1])
def test_an_unsupported_schema_version_is_refused(version: int) -> None:
    with pytest.raises(ValidationError, match="Unsupported event schema version"):
        Event.of("t", {}, producer="p", schema_version=version)


def test_an_unsupported_schema_version_is_refused_on_the_envelope() -> None:
    with pytest.raises(ValidationError, match="Unsupported event schema version"):
        EventMeta(producer="p", schema_version=2)


def test_every_supported_version_is_accepted() -> None:
    for version in SUPPORTED_EVENT_VERSIONS:
        assert EventMeta(producer="p", schema_version=version).schema_version == version


def test_a_producer_is_required() -> None:
    with pytest.raises(ValidationError):
        EventMeta()  # type: ignore[call-arg]


def test_a_topic_is_required() -> None:
    with pytest.raises(ValidationError):
        Event(meta=EventMeta(producer="p"))  # type: ignore[call-arg]


def test_an_event_survives_a_json_round_trip() -> None:
    built = Event.of("t", {"a": 1, "u": "é🎉"}, producer="p", correlation_id=uuid4())
    restored = Event.model_validate(json.loads(built.model_dump_json()))
    assert restored == built


def test_a_round_tripped_event_keeps_its_timezone() -> None:
    built = Event.of("t", {}, producer="p")
    restored = Event.model_validate(json.loads(built.model_dump_json()))
    assert restored.meta.occurred_at == built.meta.occurred_at
    assert restored.meta.occurred_at.tzinfo is not None


def test_a_round_tripped_event_keeps_its_identifiers_as_uuids() -> None:
    built = Event.of("t", {}, producer="p", correlation_id=uuid4())
    restored = Event.model_validate(json.loads(built.model_dump_json()))
    assert isinstance(restored.meta.event_id, UUID)
    assert isinstance(restored.meta.correlation_id, UUID)


def test_a_dumped_event_carries_the_fields_a_consumer_reads() -> None:
    dumped = json.loads(Event.of("t", {"a": 1}, producer="p").model_dump_json())
    assert set(dumped) == {"topic", "meta", "payload"}
    assert set(dumped["meta"]) == {
        "event_id",
        "occurred_at",
        "producer",
        "correlation_id",
        "schema_version",
    }


def test_an_event_from_a_future_version_is_refused_on_validation() -> None:
    """A consumer must reject an envelope it was not written against."""
    dumped = json.loads(Event.of("t", {}, producer="p").model_dump_json())
    dumped["meta"]["schema_version"] = 2
    with pytest.raises(ValidationError, match="Unsupported event schema version"):
        Event.model_validate(dumped)


def test_the_current_version_is_supported() -> None:
    assert CURRENT_EVENT_VERSION in SUPPORTED_EVENT_VERSIONS
