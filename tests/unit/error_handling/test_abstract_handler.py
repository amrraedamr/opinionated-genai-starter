"""What `run` and `arun` do to a failing call, independently of any SDK.

Everything a provider handler relies on lives here: how many attempts a retry
policy allows, which failures earn another one, what a caller finally sees,
and that a domain error passing through is not re-wrapped. A stub provider
stands in for an SDK so the test states the behaviour rather than one client's
exception tree.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from base.models.definitions.error_handling import FailureKind
from base.models.view_models.exceptions.provider import ProviderError
from base.tools.attempt_trail import (
    mark_attempt,
    start_attempt_trail,
    take_attempt_trail,
)
from tests.fakes import FlakyCall, RecordingCall, no_sleep
from tests.unit.error_handling.support import (
    MisMappedHandler,
    StubAuthError,
    StubBadRequestError,
    StubConflictError,
    StubConnectionError,
    StubHandler,
    StubProviderError,
    StubRateLimitError,
    StubServerError,
    StubTimeoutError,
    StubUnexpectedError,
    StubUnrelatedError,
    record_waits,
)

RETRYABLE_KINDS = (
    FailureKind.RATE_LIMIT,
    FailureKind.SERVER,
    FailureKind.TIMEOUT,
    FailureKind.CONNECTION,
)
TERMINAL_KINDS = (
    FailureKind.BAD_REQUEST,
    FailureKind.AUTH,
    FailureKind.NOT_FOUND,
    FailureKind.CONFLICT,
    FailureKind.UNKNOWN,
)


def test_a_call_that_succeeds_is_returned_untouched() -> None:
    call = RecordingCall(reply="answer")
    assert StubHandler().run(call, 1, key="v") == "answer"
    assert call.count == 1
    assert call.last == {"key": "v"}


async def test_an_awaited_call_that_succeeds_is_returned_untouched() -> None:
    call = RecordingCall(reply="answer")
    assert await StubHandler().arun(call.acall, key="v") == "answer"
    assert call.count == 1


@pytest.mark.parametrize("kind", RETRYABLE_KINDS)
def test_a_transient_failure_is_tried_again_until_it_succeeds(
    monkeypatch: pytest.MonkeyPatch, kind: FailureKind
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=2, error=StubProviderError(kind))
    assert StubHandler(retries=3).run(call) == "ok"
    assert call.attempts == 3


@pytest.mark.parametrize("kind", RETRYABLE_KINDS)
async def test_an_awaited_transient_failure_is_tried_again_until_it_succeeds(
    monkeypatch: pytest.MonkeyPatch, kind: FailureKind
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=2, error=StubProviderError(kind))
    assert await StubHandler(retries=3).arun(call.acall) == "ok"
    assert call.attempts == 3


@pytest.mark.parametrize("kind", TERMINAL_KINDS)
def test_a_failure_describing_the_request_is_not_tried_a_second_time(
    monkeypatch: pytest.MonkeyPatch, kind: FailureKind
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=5, error=StubProviderError(kind))
    with pytest.raises(ProviderError):
        StubHandler(retries=3).run(call)
    assert call.attempts == 1


@pytest.mark.parametrize("kind", TERMINAL_KINDS)
async def test_an_awaited_failure_describing_the_request_is_not_tried_again(
    monkeypatch: pytest.MonkeyPatch, kind: FailureKind
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=5, error=StubProviderError(kind))
    with pytest.raises(ProviderError):
        await StubHandler(retries=3).arun(call.acall)
    assert call.attempts == 1


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (FailureKind.BAD_REQUEST, StubBadRequestError),
        (FailureKind.AUTH, StubAuthError),
        (FailureKind.UNKNOWN, StubUnexpectedError),
    ],
)
def test_a_terminal_failure_arrives_as_the_class_its_kind_maps_to(
    kind: FailureKind, expected: type[Exception]
) -> None:
    error = StubProviderError(kind, "no")
    with pytest.raises(expected) as caught:
        StubHandler().run(RecordingCall(error=error))
    assert caught.value.provider_error is error
    assert caught.value.__cause__ is error


def test_retries_run_the_configured_number_of_extra_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=99, error=StubProviderError(FailureKind.SERVER))
    with pytest.raises(StubServerError):
        StubHandler(retries=3).run(call)
    assert call.attempts == 4


def test_no_retries_means_exactly_one_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=99, error=StubProviderError(FailureKind.SERVER))
    with pytest.raises(StubServerError) as caught:
        StubHandler(retries=0).run(call)
    assert call.attempts == 1
    assert caught.value.attempts == 1


def test_one_retry_means_exactly_two_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=99, error=StubProviderError(FailureKind.TIMEOUT))
    with pytest.raises(StubTimeoutError) as caught:
        StubHandler(retries=1).run(call)
    assert call.attempts == 2
    assert caught.value.attempts == 2


async def test_no_retries_means_exactly_one_awaited_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=99, error=StubProviderError(FailureKind.CONNECTION))
    with pytest.raises(StubConnectionError):
        await StubHandler(retries=0).arun(call.acall)
    assert call.attempts == 1


def test_an_exhausted_retry_reports_how_many_attempts_it_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    error = StubProviderError(FailureKind.RATE_LIMIT, "429")
    with pytest.raises(StubRateLimitError) as caught:
        StubHandler(retries=2).run(RecordingCall(error=error))
    assert caught.value.attempts == 3
    assert caught.value.provider_error is error
    assert "after 3 attempts" in str(caught.value)


def test_retry_false_skips_the_loop_however_many_retries_were_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=99, error=StubProviderError(FailureKind.SERVER))
    with pytest.raises(StubServerError) as caught:
        StubHandler(retries=9).run(call, retry=False)
    assert call.attempts == 1
    assert caught.value.attempts == 1


async def test_retry_false_skips_the_loop_when_awaited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=99, error=StubProviderError(FailureKind.SERVER))
    with pytest.raises(StubServerError):
        await StubHandler(retries=9).arun(call.acall, retry=False)
    assert call.attempts == 1


def test_an_exception_the_client_never_promised_becomes_the_unexpected_error() -> None:
    error = StubUnrelatedError("from somewhere else")
    with pytest.raises(StubUnexpectedError) as caught:
        StubHandler().run(RecordingCall(error=error))
    assert caught.value.provider_error is error


async def test_an_awaited_exception_the_client_never_promised_is_unexpected() -> None:
    error = ValueError("the reply did not parse")
    with pytest.raises(StubUnexpectedError) as caught:
        await StubHandler().arun(RecordingCall(error=error).acall)
    assert caught.value.provider_error is error


def test_an_unexpected_exception_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=99, error=ValueError("bad json"))
    with pytest.raises(StubUnexpectedError):
        StubHandler(retries=3).run(call)
    assert call.attempts == 1


def test_a_domain_error_raised_inside_the_call_is_not_rewrapped() -> None:
    inner = StubBadRequestError("already classified")
    with pytest.raises(StubBadRequestError) as caught:
        StubHandler().run(RecordingCall(error=inner))
    assert caught.value is inner


async def test_an_awaited_domain_error_raised_inside_the_call_is_not_rewrapped() -> (
    None
):
    inner = StubBadRequestError("already classified")
    with pytest.raises(StubBadRequestError) as caught:
        await StubHandler().arun(RecordingCall(error=inner).acall)
    assert caught.value is inner


def test_a_domain_error_passing_through_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=99, error=StubRateLimitError("x", attempts=4))
    with pytest.raises(StubRateLimitError):
        StubHandler(retries=3).run(call)
    assert call.attempts == 1


def test_a_cancellation_is_not_turned_into_a_domain_error() -> None:
    with pytest.raises(KeyboardInterrupt):
        StubHandler().run(RecordingCall(error=KeyboardInterrupt()))


async def test_an_awaited_cancellation_is_not_turned_into_a_domain_error() -> None:
    async def cancelled() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await StubHandler().arun(cancelled)


def test_the_providers_own_hint_is_waited_rather_than_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits = record_waits(monkeypatch)
    error = StubProviderError(FailureKind.RATE_LIMIT, hint=2.5)
    with pytest.raises(StubRateLimitError):
        StubHandler(retries=2).run(RecordingCall(error=error))
    assert waits == [2.5, 2.5]


async def test_the_providers_own_hint_is_awaited_rather_than_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits = record_waits(monkeypatch)
    error = StubProviderError(FailureKind.RATE_LIMIT, hint=1.25)
    with pytest.raises(StubRateLimitError):
        await StubHandler(retries=1).arun(RecordingCall(error=error).acall)
    assert waits == [1.25]


def test_backoff_grows_and_stays_under_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits = record_waits(monkeypatch)
    handler = StubHandler(retries=4, base_wait=1.0, max_wait=4.0)
    with pytest.raises(StubServerError):
        handler.run(RecordingCall(error=StubProviderError(FailureKind.SERVER)))
    assert len(waits) == 4
    assert all(0.0 <= wait <= 4.0 for wait in waits)


def test_a_zero_base_wait_sleeps_for_nothing_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits = record_waits(monkeypatch)
    handler = StubHandler(retries=3, base_wait=0.0)
    with pytest.raises(StubServerError):
        handler.run(RecordingCall(error=StubProviderError(FailureKind.SERVER)))
    assert waits == [0.0, 0.0, 0.0]


def test_the_last_attempt_is_not_followed_by_a_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits = record_waits(monkeypatch)
    with pytest.raises(StubServerError):
        StubHandler(retries=2, base_wait=0.0).run(
            RecordingCall(error=StubProviderError(FailureKind.SERVER))
        )
    assert len(waits) == 2


def test_the_defaults_are_three_retries_and_half_a_second() -> None:
    handler = StubHandler()
    assert (handler.DEFAULT_RETRIES, handler.DEFAULT_BASE_WAIT) == (3, 0.5)
    assert handler.DEFAULT_MAX_WAIT == 10.0


def test_a_mapping_that_cannot_report_attempts_is_caught_by_an_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    with pytest.raises(AssertionError, match="non-retry-exhausted"):
        MisMappedHandler(retries=1).run(
            RecordingCall(error=StubProviderError(FailureKind.SERVER))
        )


def test_wrap_exhausted_refuses_a_class_that_cannot_report_attempts() -> None:
    with pytest.raises(AssertionError, match="StubBadRequestError"):
        MisMappedHandler().wrap_exhausted(StubProviderError(FailureKind.SERVER), 2)


def test_a_handler_that_fails_while_classifying_does_not_hide_its_own_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenHandler(StubHandler):
        def classify(self, error: StubProviderError) -> tuple[type[Exception], bool]:
            raise RuntimeError("the map is wrong")

    no_sleep(monkeypatch)
    with pytest.raises(RuntimeError, match="the map is wrong"):
        BrokenHandler().run(RecordingCall(error=StubProviderError()))


async def test_a_handler_that_fails_while_classifying_an_awaited_call_says_so() -> None:
    class BrokenHandler(StubHandler):
        def classify(self, error: StubProviderError) -> tuple[type[Exception], bool]:
            raise RuntimeError("the map is wrong")

    with pytest.raises(RuntimeError, match="the map is wrong"):
        await BrokenHandler().arun(RecordingCall(error=StubProviderError()).acall)


def test_a_handler_that_fails_while_reading_a_retry_hint_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenHandler(StubHandler):
        def retry_after(self, error: StubProviderError) -> float | None:
            raise RuntimeError("the header is wrong")

    no_sleep(monkeypatch)
    with pytest.raises(RuntimeError, match="the header is wrong"):
        BrokenHandler(retries=2).run(
            RecordingCall(error=StubProviderError(FailureKind.SERVER))
        )


def test_a_handler_that_fails_while_wrapping_says_so() -> None:
    class BrokenHandler(StubHandler):
        def wrap(
            self, error: StubProviderError, domain_cls: type[Exception]
        ) -> Exception:
            raise RuntimeError("cannot build the domain error")

    with pytest.raises(RuntimeError, match="cannot build the domain error"):
        BrokenHandler().run(
            RecordingCall(error=StubProviderError(FailureKind.BAD_REQUEST))
        )


def test_an_unhandled_failure_the_caller_drove_becomes_a_domain_error() -> None:
    error = StubProviderError(FailureKind.CONFLICT, "duplicate")
    wrapped = StubHandler().as_domain_error(error)
    assert type(wrapped).__name__ == "StubConflictError"
    assert wrapped.provider_error is error


def test_an_unhandled_transient_failure_is_reported_as_one_attempt() -> None:
    error = StubProviderError(FailureKind.SERVER, "503")
    wrapped = StubHandler().as_domain_error(error)
    assert isinstance(wrapped, StubServerError)
    assert wrapped.attempts == 1


def test_an_unhandled_foreign_exception_becomes_the_unexpected_error() -> None:
    error = ValueError("nothing to do with the client")
    wrapped = StubHandler().as_domain_error(error)
    assert isinstance(wrapped, StubUnexpectedError)
    assert wrapped.provider_error is error


def test_as_domain_error_counts_no_attempt_because_it_ran_none() -> None:
    start_attempt_trail()
    mark_attempt()
    StubHandler().as_domain_error(StubProviderError(FailureKind.SERVER))
    assert take_attempt_trail().attempts == 1


def test_the_trail_counts_every_attempt_a_call_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    call = FlakyCall(failures=2, error=StubProviderError(FailureKind.SERVER, "503"))
    StubHandler(retries=3).run(call)
    recorded = take_attempt_trail()
    assert recorded.attempts == 3
    assert recorded.errors == ("StubProviderError: 503", "StubProviderError: 503")


def test_a_call_that_worked_first_time_leaves_no_failures() -> None:
    StubHandler().run(RecordingCall(reply="fine"))
    recorded = take_attempt_trail()
    assert (recorded.attempts, recorded.errors) == (1, ())


def test_the_trail_survives_a_call_that_exhausted_its_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    error = StubProviderError(FailureKind.CONNECTION, "refused")
    with pytest.raises(StubConnectionError):
        StubHandler(retries=2).run(RecordingCall(error=error))
    recorded = take_attempt_trail()
    assert recorded.attempts == 3
    assert recorded.errors == ("StubProviderError: refused",) * 3


def test_a_terminal_failure_leaves_the_one_attempt_it_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    with pytest.raises(StubAuthError):
        StubHandler(retries=3).run(
            RecordingCall(error=StubProviderError(FailureKind.AUTH, "bad key"))
        )
    recorded = take_attempt_trail()
    assert (recorded.attempts, recorded.errors) == (
        1,
        ("StubProviderError: bad key",),
    )


def test_a_domain_error_passing_through_is_not_added_to_the_trail() -> None:
    with pytest.raises(StubBadRequestError):
        StubHandler().run(RecordingCall(error=StubBadRequestError("inner verdict")))
    assert take_attempt_trail().errors == ()


def test_one_handler_serving_concurrent_calls_keeps_their_trails_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    handler = StubHandler(retries=4)

    async def call(failures: int) -> tuple[int, int]:
        flaky = FlakyCall(
            failures=failures, error=StubProviderError(FailureKind.TIMEOUT)
        )
        await handler.arun(flaky.acall)
        return failures, take_attempt_trail().attempts

    async def main() -> list[tuple[int, int]]:
        return await asyncio.gather(*(call(n) for n in (0, 1, 2, 3)))

    assert asyncio.run(main()) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_one_handler_serving_concurrent_calls_classifies_each_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_sleep(monkeypatch)
    handler = StubHandler(retries=1)

    async def call(kind: FailureKind) -> type[Exception]:
        try:
            await handler.arun(RecordingCall(error=StubProviderError(kind)).acall)
        except ProviderError as exc:
            return type(exc)
        raise AssertionError("the call should have failed")

    async def main() -> list[type[Exception]]:
        return await asyncio.gather(
            *(
                call(kind)
                for kind in (
                    FailureKind.AUTH,
                    FailureKind.SERVER,
                    FailureKind.CONFLICT,
                    FailureKind.TIMEOUT,
                )
            )
        )

    assert asyncio.run(main()) == [
        StubAuthError,
        StubServerError,
        StubConflictError,
        StubTimeoutError,
    ]


def test_arguments_reach_the_call_unchanged() -> None:
    def add(first: int, second: int, *, scale: int = 1) -> int:
        return (first + second) * scale

    assert StubHandler().run(add, 2, 3, scale=10) == 50


async def test_arguments_reach_an_awaited_call_unchanged() -> None:
    async def add(first: int, second: int, *, scale: int = 1) -> int:
        return (first + second) * scale

    assert await StubHandler().arun(add, 2, 3, scale=10) == 50


def test_run_and_arun_cannot_be_overridden_by_a_subclass() -> None:
    assert getattr(StubHandler.run, "__final__", False)
    assert getattr(StubHandler.arun, "__final__", False)
    assert getattr(StubHandler.as_domain_error, "__final__", False)


def test_a_handler_without_the_abstract_methods_cannot_be_instantiated() -> None:
    from base.services.error_handling import AbstractErrorHandler

    class Incomplete(AbstractErrorHandler[Any]):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]
