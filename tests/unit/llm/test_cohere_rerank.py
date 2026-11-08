"""The Cohere rerank adapter: what it sends, and what a failure becomes.

The response shaping is pinned in `test_embedding_and_rerank.py` beside the
abstract service. What is pinned here is the half that only this adapter owns:
the request it builds, the clients it builds lazily so a service that never
reranks never holds one, and the errors the SDK raises becoming the rerank
family rather than escaping as Cohere's own.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from cohere.errors import BadRequestError, TooManyRequestsError, UnauthorizedError

from base.models.view_models.exceptions.rerank import (
    RerankAuthError,
    RerankBadRequestError,
    RerankRateLimitError,
    RerankUnexpectedError,
)
from base.services.reranking.cohere import CohereRerankService
from tests.fakes import FlakyCall, RecordingCall, no_sleep

DOCUMENTS = ["zero", "one", "two"]


def reply(*scores: float) -> SimpleNamespace:
    """A Cohere rerank response, best first."""
    return SimpleNamespace(
        results=[
            SimpleNamespace(index=index, relevance_score=score)
            for index, score in enumerate(scores)
        ],
        meta=SimpleNamespace(
            tokens=SimpleNamespace(input_tokens=11, output_tokens=0), cached_tokens=0
        ),
    )


def build(model: str = "rerank-v4.0-fast") -> CohereRerankService:
    """The adapter without the singleton, as the sibling tests build it."""
    config: Any = SimpleNamespace(cohere_rerank_model=model, cohere_api_key="k")
    service = object.__new__(CohereRerankService)
    CohereRerankService.__init__(service, config)
    return service


def answering(service: CohereRerankService, call: Any) -> None:
    """Point both clients at one fake, so sync and async send the same request."""
    service._sync_client = SimpleNamespace(rerank=call)  # type: ignore[assignment]
    service._async_client = SimpleNamespace(rerank=call.acall)  # type: ignore[assignment]


def test_a_client_is_built_on_first_use_and_then_kept() -> None:
    service = build()

    first = service._client

    assert first is service._client
    assert service._async_client is None


async def test_an_async_client_is_built_on_first_use_and_then_kept() -> None:
    service = build()

    first = service._aclient

    assert first is service._aclient
    assert service._sync_client is None


def test_sdk_retries_are_off_because_the_error_handler_owns_the_policy() -> None:
    """Two layers of backoff would multiply the wait."""
    service = build()

    assert service._client._client_wrapper.get_max_retries() == 0
    assert service._aclient._client_wrapper.get_max_retries() == 0


def test_the_configured_model_and_the_query_are_what_is_sent() -> None:
    service = build("rerank-v3.5")
    call = RecordingCall(reply(0.9, 0.5, 0.1))
    answering(service, call)

    service.rerank("which?", DOCUMENTS)

    assert call.last["model"] == "rerank-v3.5"
    assert call.last["query"] == "which?"
    assert call.last["documents"] == DOCUMENTS


def test_asking_for_no_cut_off_asks_for_every_document() -> None:
    """Cohere requires `top_n`, so the whole list stands in for "all of them"."""
    service = build()
    call = RecordingCall(reply(0.9, 0.5, 0.1))
    answering(service, call)

    service.rerank("which?", DOCUMENTS)

    assert call.last["top_n"] == 3


def test_a_cut_off_is_sent_as_it_was_asked_for() -> None:
    service = build()
    call = RecordingCall(reply(0.9))
    answering(service, call)

    service.rerank("which?", DOCUMENTS, top_n=1)

    assert call.last["top_n"] == 1


def test_a_cut_off_of_zero_is_read_as_every_document() -> None:
    """Pinned as it stands: `0` is falsy, so it cannot mean "none"."""
    service = build()
    call = RecordingCall(reply(0.9, 0.5, 0.1))
    answering(service, call)

    service.rerank("which?", DOCUMENTS, top_n=0)

    assert call.last["top_n"] == 3


async def test_the_async_surface_builds_the_same_request() -> None:
    service = build()
    call = RecordingCall(reply(0.9, 0.5, 0.1))
    answering(service, call)

    await service.arerank("which?", DOCUMENTS, top_n=2)

    assert call.last == {
        "model": "rerank-v4.0-fast",
        "query": "which?",
        "documents": DOCUMENTS,
        "top_n": 2,
    }


def test_the_scored_results_are_mapped_back_to_their_documents() -> None:
    service = build()
    answering(service, RecordingCall(reply(0.9, 0.5, 0.1)))

    results = service.rerank("which?", DOCUMENTS)

    assert [(item.index, item.document) for item in results] == [
        (0, "zero"),
        (1, "one"),
        (2, "two"),
    ]


async def test_the_async_surface_shapes_the_reply_the_same_way() -> None:
    service = build()
    answering(service, RecordingCall(reply(0.9, 0.5, 0.1)))

    results = await service.arerank("which?", DOCUMENTS)

    assert [item.score for item in results] == [0.9, 0.5, 0.1]


def test_a_refused_request_becomes_a_rerank_error_rather_than_cohere_s_own() -> None:
    service = build()
    answering(service, RecordingCall(error=BadRequestError("too many documents", {})))

    with pytest.raises(RerankBadRequestError, match="too many documents"):
        service.rerank("which?", DOCUMENTS)


async def test_a_refused_async_request_becomes_a_rerank_error_too() -> None:
    service = build()
    answering(service, RecordingCall(error=UnauthorizedError("bad key", {})))

    with pytest.raises(RerankAuthError):
        await service.arerank("which?", DOCUMENTS)


def test_a_request_that_will_never_work_is_not_repeated() -> None:
    service = build()
    call = RecordingCall(error=BadRequestError("malformed", {}))
    answering(service, call)

    with pytest.raises(RerankBadRequestError):
        service.rerank("which?", DOCUMENTS)
    assert call.count == 1


def test_a_rate_limit_is_retried_and_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = build()
    call = FlakyCall(2, TooManyRequestsError("slow down", {}), reply(0.9, 0.5, 0.1))
    answering(service, call)

    results = service.rerank("which?", DOCUMENTS)

    assert call.attempts == 3
    assert len(results) == 3


async def test_a_rate_limit_is_retried_on_the_async_surface_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = build()
    call = FlakyCall(1, TooManyRequestsError("slow down", {}), reply(0.9))
    answering(service, call)

    await service.arerank("which?", DOCUMENTS, top_n=1)

    assert call.attempts == 2


def test_retries_run_out_rather_than_looping(monkeypatch: pytest.MonkeyPatch) -> None:
    no_sleep(monkeypatch)
    service = build()
    call = RecordingCall(error=TooManyRequestsError("slow down", {}))
    answering(service, call)

    with pytest.raises(RerankRateLimitError, match="after 4 attempts"):
        service.rerank("which?", DOCUMENTS)
    assert call.count == 4


def test_a_reply_the_adapter_cannot_read_becomes_a_rerank_error() -> None:
    """Shaping happens inside the retried body, so a bad reply is not raw."""
    service = build()
    answering(service, RecordingCall(SimpleNamespace(results="not a list", meta=None)))

    with pytest.raises(RerankUnexpectedError):
        service.rerank("which?", DOCUMENTS)


def test_something_the_sdk_never_promised_becomes_a_rerank_error() -> None:
    service = build()
    answering(service, RecordingCall(error=RuntimeError("socket closed")))

    with pytest.raises(RerankUnexpectedError, match="socket closed"):
        service.rerank("which?", DOCUMENTS)


def test_an_empty_request_never_reaches_the_client() -> None:
    service = build()
    call = RecordingCall(reply())
    answering(service, call)

    assert service.rerank("   ", DOCUMENTS) == []
    assert service.rerank("which?", []) == []
    assert call.count == 0
