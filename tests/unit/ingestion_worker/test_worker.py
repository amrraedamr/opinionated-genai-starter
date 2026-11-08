"""The composition root: what it subscribes, in what order, and how it stops.

This service has no HTTP surface, so `arun` is the whole of its startup —
every handler it owns, wrapped in the inbox, on the group every replica
shares. `start` is the shutdown half: it must stop consuming before the
database it was writing claims into goes away.
"""

from __future__ import annotations

import signal
from typing import Any

import pytest

from base.models.definitions.messaging import InboxState
from base.services.broker.memory import InMemoryBroker
from contracts.v1 import chat, identity
from ingestion_worker import handlers, worker
from tests.unit.ingestion_worker.conftest import (
    RecordingBroker,
    conversation_created,
    row_count,
    set_claim,
    stored,
    user_registered,
)

# What it subscribes


async def test_it_subscribes_every_topic_this_worker_owns(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    del worker_settings
    await worker.arun(broker)
    assert broker.topics == [identity.USER_REGISTERED, chat.CONVERSATION_CREATED]


async def test_it_subscribes_nothing_else(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    del worker_settings
    await worker.arun(broker)
    assert len(broker.subscriptions) == 2


async def test_every_subscription_joins_the_group_the_settings_name(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    """One group across replicas is what makes an event handled once, not once each."""
    del worker_settings
    await worker.arun(broker)
    assert broker.groups == ["ingestion_worker", "ingestion_worker"]


async def test_the_group_is_the_configured_one_rather_than_the_service_name(
    make_worker_settings: Any, broker: RecordingBroker
) -> None:
    make_worker_settings(consumer_group="ingestion_worker_canary")
    await worker.arun(broker)
    assert set(broker.groups) == {"ingestion_worker_canary"}


async def test_nothing_is_subscribed_after_consuming_has_begun(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    """A transport reads its subscriptions on start, so a late one hears nothing."""
    del worker_settings
    await worker.arun(broker)
    assert broker.trail == ["subscribe", "subscribe", "start"]


async def test_it_answers_with_the_broker_it_was_given(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    del worker_settings
    assert await worker.arun(broker) is broker


async def test_it_builds_the_transport_the_settings_name_when_given_none(
    worker_settings: Any,
) -> None:
    del worker_settings
    built = await worker.arun()
    try:
        assert isinstance(built, InMemoryBroker)
    finally:
        await built.astop()


async def test_it_publishes_nothing_of_its_own(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    """A consumer that produced would need an outbox of its own."""
    del worker_settings
    await worker.arun(broker)
    assert broker.published == []


# What it subscribed is the handler behind the inbox


@pytest.mark.parametrize("topic", [identity.USER_REGISTERED, chat.CONVERSATION_CREATED])
async def test_the_handler_it_subscribed_is_not_the_bare_one(
    worker_settings: Any, broker: RecordingBroker, topic: str
) -> None:
    del worker_settings
    await worker.arun(broker)
    assert broker.handler_for(topic) not in (
        handlers.ahandle_user_registered,
        handlers.ahandle_conversation_created,
    )


async def test_a_delivery_reaches_the_handler_and_commits_a_claim(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    del worker_settings
    await worker.arun(broker)
    delivered = user_registered()
    await broker.handler_for(identity.USER_REGISTERED)(delivered)
    assert handlers.processed_count() == 1
    assert (await stored(delivered)).state == InboxState.PROCESSED


async def test_the_same_delivery_twice_reaches_the_handler_once(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    del worker_settings
    await worker.arun(broker)
    delivered = conversation_created()
    wrapped = broker.handler_for(chat.CONVERSATION_CREATED)
    await wrapped(delivered)
    await wrapped(delivered)
    assert handlers.processed_count() == 1


async def test_both_topics_claim_into_one_inbox(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    """One table, so a restart sees everything this worker has ever finished."""
    del worker_settings
    await worker.arun(broker)
    await broker.handler_for(identity.USER_REGISTERED)(user_registered())
    await broker.handler_for(chat.CONVERSATION_CREATED)(conversation_created())
    assert handlers.processed_count() == 2
    assert await row_count() == 2


async def test_two_topics_sharing_one_event_id_are_handled_once(
    worker_settings: Any, broker: RecordingBroker
) -> None:
    """The claim keys on the id alone, so a producer reusing one costs a delivery."""
    del worker_settings
    await worker.arun(broker)
    first = user_registered()
    second = conversation_created()
    second.meta.event_id = first.meta.event_id
    await broker.handler_for(identity.USER_REGISTERED)(first)
    await broker.handler_for(chat.CONVERSATION_CREATED)(second)
    assert handlers.processed_count() == 1
    assert await row_count() == 1


async def test_a_failing_handler_is_reported_to_the_transport(
    worker_settings: Any, broker: RecordingBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A swallowed failure would be acked and never redelivered."""
    del worker_settings

    async def failing(incoming: Any) -> None:
        raise RuntimeError("handler failed")

    monkeypatch.setattr(handlers, "ahandle_user_registered", failing)
    await worker.arun(broker)
    delivered = user_registered()
    with pytest.raises(RuntimeError, match="handler failed"):
        await broker.handler_for(identity.USER_REGISTERED)(delivered)
    assert (await stored(delivered)).state == InboxState.FAILED


# The claim timeout it was configured with


async def test_the_claim_timeout_comes_from_the_settings(
    make_worker_settings: Any, broker: RecordingBroker
) -> None:
    make_worker_settings(inbox_claim_timeout_seconds=30)
    await worker.arun(broker)
    wrapped = broker.handler_for(identity.USER_REGISTERED)
    delivered = user_registered()
    await wrapped(delivered)
    await set_claim(delivered, state=InboxState.PROCESSING, age_seconds=600)
    await wrapped(delivered)
    assert handlers.processed_count() == 2


async def test_a_claim_timeout_longer_than_the_outage_holds_the_lock(
    make_worker_settings: Any, broker: RecordingBroker
) -> None:
    make_worker_settings(inbox_claim_timeout_seconds=3_600)
    await worker.arun(broker)
    wrapped = broker.handler_for(identity.USER_REGISTERED)
    delivered = user_registered()
    await wrapped(delivered)
    await set_claim(delivered, state=InboxState.PROCESSING, age_seconds=600)
    await wrapped(delivered)
    assert handlers.processed_count() == 1


# Shutdown


def run_until_signalled(monkeypatch: pytest.MonkeyPatch, signum: int) -> list[str]:
    """Drive `start` through a real signal and report the order it released things."""
    trail: list[str] = []

    class Broker:
        async def astop(self) -> None:
            trail.append("transport stopped")

    class Sql:
        async def adispose(self) -> None:
            trail.append("database released")

    async def arun() -> Broker:
        trail.append("consuming")
        signal.raise_signal(signum)
        return Broker()

    monkeypatch.setattr(worker, "arun", arun)
    monkeypatch.setattr(worker, "SQLService", Sql)
    worker.start()
    return trail


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_the_worker_shuts_down_on_a_signal(
    monkeypatch: pytest.MonkeyPatch, signum: int
) -> None:
    """A container sends SIGTERM and a developer sends SIGINT; both must land."""
    assert run_until_signalled(monkeypatch, signum)[0] == "consuming"


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_consuming_stops_before_the_database_is_released(
    monkeypatch: pytest.MonkeyPatch, signum: int
) -> None:
    """A handler still running would lose the session it was writing its claim in."""
    assert run_until_signalled(monkeypatch, signum) == [
        "consuming",
        "transport stopped",
        "database released",
    ]
