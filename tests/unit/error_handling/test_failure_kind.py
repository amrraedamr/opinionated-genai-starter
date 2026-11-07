"""Which failures are worth trying again, stated once for every provider.

`RETRYABLE_FAILURES` is the single place that decision is made, so a handler
that classifies correctly cannot retry the wrong thing. Pinning the exact set
is what stops a kind quietly joining or leaving it.
"""

from base.models.definitions.error_handling import RETRYABLE_FAILURES, FailureKind


def test_the_nine_failure_kinds_are_the_whole_vocabulary() -> None:
    assert {kind.value for kind in FailureKind} == {
        "rate_limit",
        "server",
        "timeout",
        "connection",
        "bad_request",
        "auth",
        "not_found",
        "conflict",
        "unknown",
    }


def test_exactly_four_kinds_are_retried() -> None:
    assert RETRYABLE_FAILURES == frozenset(
        {
            FailureKind.RATE_LIMIT,
            FailureKind.SERVER,
            FailureKind.TIMEOUT,
            FailureKind.CONNECTION,
        }
    )


def test_a_failure_describing_the_request_is_never_retried() -> None:
    for kind in (
        FailureKind.BAD_REQUEST,
        FailureKind.AUTH,
        FailureKind.NOT_FOUND,
        FailureKind.CONFLICT,
        FailureKind.UNKNOWN,
    ):
        assert kind not in RETRYABLE_FAILURES


def test_every_kind_is_either_retryable_or_not_and_none_is_missed() -> None:
    assert RETRYABLE_FAILURES < set(FailureKind)
    assert len(RETRYABLE_FAILURES) + 5 == len(FailureKind)


def test_a_kind_compares_equal_to_its_wire_string() -> None:
    assert FailureKind.RATE_LIMIT == "rate_limit"
    assert FailureKind("server") is FailureKind.SERVER


def test_the_retryable_set_cannot_be_mutated_by_a_caller() -> None:
    assert isinstance(RETRYABLE_FAILURES, frozenset)
    assert not hasattr(RETRYABLE_FAILURES, "add")
