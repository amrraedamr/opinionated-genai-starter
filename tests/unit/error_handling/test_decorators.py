"""The four decorators every service method is covered by, and what they imply.

`retrying` repeats a body; `classifying` deliberately does not, because the
statement it carries may already have landed. Both turn a client failure into
the domain's own error. The case this file exists for is a decorated method
calling a decorated sibling: the inner verdict must survive, rather than being
re-read as something nobody anticipated. That was a real defect, fixed at
every call site, and only a test keeps it fixed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from base.models.definitions.error_handling import FailureKind
from base.services.error_handling import (
    AbstractErrorHandler,
    aclassifying,
    aretrying,
    classifying,
    retrying,
)
from base.tools.attempt_trail import take_attempt_trail
from tests.fakes import no_sleep
from tests.unit.error_handling.support import (
    StubAuthError,
    StubBadRequestError,
    StubConflictError,
    StubHandler,
    StubProviderError,
    StubServerError,
    StubUnexpectedError,
)


class Service:
    """A service shaped like the real ones: one handler, decorated methods."""

    def __init__(self, handler: AbstractErrorHandler[Any] | None = None) -> None:
        self._error_handler = handler or StubHandler(retries=2, base_wait=0.0)
        self.calls: list[str] = []
        self.failures: int = 0
        self.error: Exception = StubProviderError(FailureKind.SERVER)

    @retrying
    def fetch(self, token: str = "t") -> str:
        self.calls.append("fetch")
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return f"fetched {token}"

    @aretrying
    async def afetch(self, token: str = "t") -> str:
        self.calls.append("afetch")
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return f"fetched {token}"

    @classifying
    def execute(self, statement: str = "s") -> str:
        self.calls.append("execute")
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return f"executed {statement}"

    @aclassifying
    async def aexecute(self, statement: str = "s") -> str:
        self.calls.append("aexecute")
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return f"executed {statement}"

    @retrying
    def outer(self) -> str:
        self.calls.append("outer")
        return self.fetch()

    @aretrying
    async def aouter(self) -> str:
        self.calls.append("aouter")
        return await self.afetch()

    @classifying
    def outer_classifying(self) -> str:
        self.calls.append("outer_classifying")
        return self.execute()

    @retrying
    def stream(self) -> Iterator[str]:
        self.calls.append("stream")
        yield "chunk"
        raise self.error


def test_a_decorated_method_returns_its_own_result() -> None:
    assert Service().fetch("abc") == "fetched abc"


async def test_a_decorated_async_method_returns_its_own_result() -> None:
    assert await Service().afetch("abc") == "fetched abc"


def test_a_decorated_method_keeps_its_name_and_wrapped_function() -> None:
    assert Service.fetch.__name__ == "fetch"
    assert Service.aexecute.__name__ == "aexecute"
    assert Service.fetch.__wrapped__.__qualname__.endswith("Service.fetch")


def test_retrying_repeats_a_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.failures = 2
    assert service.fetch() == "fetched t"
    assert service.calls == ["fetch", "fetch", "fetch"]


async def test_aretrying_repeats_a_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.failures = 2
    assert await service.afetch() == "fetched t"
    assert service.calls == ["afetch"] * 3


@pytest.mark.parametrize(
    "kind",
    [
        FailureKind.RATE_LIMIT,
        FailureKind.SERVER,
        FailureKind.TIMEOUT,
        FailureKind.CONNECTION,
    ],
)
def test_retrying_repeats_exactly_the_retryable_kinds(
    monkeypatch: pytest.MonkeyPatch, kind: FailureKind
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.error = StubProviderError(kind)
    service.failures = 1
    assert service.fetch() == "fetched t"
    assert len(service.calls) == 2


@pytest.mark.parametrize(
    "kind",
    [
        FailureKind.BAD_REQUEST,
        FailureKind.AUTH,
        FailureKind.NOT_FOUND,
        FailureKind.CONFLICT,
        FailureKind.UNKNOWN,
    ],
)
def test_retrying_reraises_a_non_retryable_kind_with_no_second_attempt(
    monkeypatch: pytest.MonkeyPatch, kind: FailureKind
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.error = StubProviderError(kind)
    service.failures = 99
    with pytest.raises(Exception) as caught:
        service.fetch()
    assert service.calls == ["fetch"]
    assert not isinstance(caught.value, StubServerError)


def test_retrying_stops_after_the_configured_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service(StubHandler(retries=2, base_wait=0.0))
    service.failures = 99
    with pytest.raises(StubServerError) as caught:
        service.fetch()
    assert service.calls == ["fetch"] * 3
    assert caught.value.attempts == 3


async def test_aretrying_stops_after_the_configured_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service(StubHandler(retries=1, base_wait=0.0))
    service.failures = 99
    with pytest.raises(StubServerError) as caught:
        await service.afetch()
    assert service.calls == ["afetch"] * 2
    assert caught.value.attempts == 2


def test_classifying_does_not_repeat_a_transient_failure_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.failures = 99
    with pytest.raises(StubServerError) as caught:
        service.execute()
    assert service.calls == ["execute"]
    assert caught.value.attempts == 1


async def test_aclassifying_does_not_repeat_a_transient_failure_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.failures = 99
    with pytest.raises(StubServerError) as caught:
        await service.aexecute()
    assert service.calls == ["aexecute"]
    assert caught.value.attempts == 1


def test_classifying_still_turns_a_client_failure_into_a_domain_error() -> None:
    service = Service()
    service.error = StubProviderError(FailureKind.AUTH, "bad key")
    service.failures = 1
    with pytest.raises(StubAuthError):
        service.execute()


def test_classifying_repeats_when_the_caller_asks_it_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.failures = 2
    assert service.execute(retry=True) == "executed s"
    assert service.calls == ["execute"] * 3


async def test_aclassifying_repeats_when_the_caller_asks_it_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.failures = 2
    assert await service.aexecute(retry=True) == "executed s"
    assert service.calls == ["aexecute"] * 3


def test_retrying_can_be_told_not_to_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.failures = 99
    with pytest.raises(StubServerError):
        service.fetch(retry=False)
    assert service.calls == ["fetch"]


def test_classifying_does_not_pass_its_retry_flag_to_the_method() -> None:
    service = Service()
    assert service.execute("my statement", retry=True) == "executed my statement"


def test_an_inner_domain_error_reaches_the_outer_caller_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.error = StubProviderError(FailureKind.BAD_REQUEST, "malformed")
    service.failures = 99
    with pytest.raises(StubBadRequestError) as caught:
        service.outer()
    assert not isinstance(caught.value, StubUnexpectedError)
    assert caught.value.provider_error is service.error


async def test_an_inner_domain_error_reaches_the_outer_awaiting_caller_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.error = StubProviderError(FailureKind.AUTH, "bad key")
    service.failures = 99
    with pytest.raises(StubAuthError) as caught:
        await service.aouter()
    assert not isinstance(caught.value, StubUnexpectedError)


def test_an_exhausted_inner_retry_is_not_counted_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service(StubHandler(retries=1, base_wait=0.0))
    service.failures = 99
    with pytest.raises(StubServerError) as caught:
        service.outer()
    assert caught.value.attempts == 2
    assert service.calls == ["outer", "fetch", "fetch"]


def test_a_classifying_outer_does_not_rewrap_a_classifying_inner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.error = StubProviderError(FailureKind.CONFLICT, "duplicate")
    service.failures = 99
    with pytest.raises(StubConflictError) as caught:
        service.outer_classifying()
    assert caught.value.provider_error is service.error


def test_an_outer_retry_does_not_repeat_after_an_inner_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service()
    service.error = StubProviderError(FailureKind.BAD_REQUEST)
    service.failures = 99
    with pytest.raises(StubBadRequestError):
        service.outer()
    assert service.calls == ["outer", "fetch"]


def test_an_outer_retry_does_not_repeat_after_an_inner_exhausted_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service(StubHandler(retries=2, base_wait=0.0))
    service.failures = 99
    with pytest.raises(StubServerError):
        service.outer()
    assert service.calls == ["outer", "fetch", "fetch", "fetch"]


def test_a_domain_error_from_another_family_also_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store handler wrapping a model call must not re-read the model's verdict."""
    from base.models.view_models.exceptions.llm import LLMRateLimitError

    no_sleep(monkeypatch)
    service = Service()
    service.error = LLMRateLimitError("throttled upstream", attempts=4)
    service.failures = 99
    with pytest.raises(LLMRateLimitError):
        service.fetch()
    assert service.calls == ["fetch"]


def test_a_generator_body_never_reaches_the_handler() -> None:
    """Calling a decorated generator does no work, so its failure escapes raw."""
    service = Service()
    service.error = StubProviderError(FailureKind.SERVER, "never classified")
    produced = service.stream()
    assert service.calls == []
    assert next(produced) == "chunk"
    with pytest.raises(StubProviderError):
        next(produced)


async def test_an_async_generator_cannot_be_awaited_through_the_handler() -> None:
    """`arun` awaits what the method returned, and an async generator is not."""

    class Streaming:
        def __init__(self) -> None:
            self._error_handler = StubHandler(retries=0)

        @aretrying
        async def astream(self) -> AsyncIterator[str]:
            yield "chunk"

    with pytest.raises(StubUnexpectedError) as caught:
        await Streaming().astream()
    assert isinstance(caught.value.provider_error, TypeError)


def test_the_decorator_reads_the_handler_off_the_instance_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    service = Service(StubHandler(retries=5, base_wait=0.0))
    service.failures = 99
    service._error_handler = StubHandler(retries=0)
    with pytest.raises(StubServerError) as caught:
        service.fetch()
    assert service.calls == ["fetch"]
    assert caught.value.attempts == 1


def test_a_service_without_a_handler_fails_loudly() -> None:
    class Bare:
        @retrying
        def fetch(self) -> str:
            return "never reached"

    with pytest.raises(AttributeError, match="_error_handler"):
        Bare().fetch()


def test_two_services_sharing_one_handler_do_not_share_a_trail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    handler = StubHandler(retries=3, base_wait=0.0)

    async def run(failures: int) -> int:
        service = Service(handler)
        service.failures = failures
        await service.afetch()
        return take_attempt_trail().attempts

    async def main() -> list[int]:
        return await asyncio.gather(*(run(n) for n in (0, 2, 1)))

    assert asyncio.run(main()) == [1, 3, 2]


def test_a_nested_call_restarts_the_trail_the_outer_one_began(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every `run` starts a fresh trail, so the inner call's is what remains."""
    no_sleep(monkeypatch)
    service = Service(StubHandler(retries=3, base_wait=0.0))
    service.failures = 2
    service.outer()
    assert service.calls == ["outer", "fetch", "fetch", "fetch"]
    assert take_attempt_trail().attempts == 3
