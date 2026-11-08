"""The guardrail seam: verdicts, the pipeline, and what a block raises.

The pipeline's rules are the whole contract callers rely on — the first block
wins, redactions compose in order, a guardrail is consulted only at the stages
it declares — so each one is pinned here against stand-in guardrails.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from base.models.definitions.errors import GuardrailErrors
from base.models.definitions.guardrails import GuardrailOutcome, GuardrailStage
from base.models.view_models.guardrails import GuardrailReport, GuardrailVerdict
from base.services.guardrails import AbstractGuardrail, GuardrailBlocked
from base.services.guardrails.pipeline import GuardrailPipeline

ALL_STAGES = frozenset(GuardrailStage)


class Fixed(AbstractGuardrail):
    """Answers one verdict, and records what it was shown."""

    def __init__(
        self,
        name: str,
        verdict: GuardrailVerdict,
        stages: frozenset[GuardrailStage] = ALL_STAGES,
    ) -> None:
        self._label = name
        self._verdict = verdict
        self._declared = stages
        self.seen: list[str] = []

    def _name(self) -> str:
        return self._label

    def _stages(self) -> frozenset[GuardrailStage]:
        return self._declared

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        self.seen.append(content)
        return self._verdict

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        self.seen.append(content)
        return self._verdict


class Upper(AbstractGuardrail):
    """Redacts by upper-casing, so composition order is visible."""

    def _name(self) -> str:
        return "upper"

    def _stages(self) -> frozenset[GuardrailStage]:
        return ALL_STAGES

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return GuardrailVerdict.redact(content.upper())

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return GuardrailVerdict.redact(content.upper())


class Suffix(AbstractGuardrail):
    def __init__(self, suffix: str) -> None:
        self._suffix = suffix

    def _name(self) -> str:
        return f"suffix-{self._suffix}"

    def _stages(self) -> frozenset[GuardrailStage]:
        return ALL_STAGES

    def _check(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return GuardrailVerdict.redact(content + self._suffix)

    async def _acheck(self, content: str, stage: GuardrailStage) -> GuardrailVerdict:
        return GuardrailVerdict.redact(content + self._suffix)


def allow(name: str = "allow", **kwargs: frozenset[GuardrailStage]) -> Fixed:
    return Fixed(name, GuardrailVerdict.allow(), **kwargs)


def block(name: str = "block", **kwargs: frozenset[GuardrailStage]) -> Fixed:
    return Fixed(name, GuardrailVerdict.block("not allowed", category="test"), **kwargs)


# Verdicts


def test_an_allow_needs_nothing_else() -> None:
    assert GuardrailVerdict.allow().outcome == GuardrailOutcome.ALLOW


def test_a_redaction_without_replacement_content_is_refused() -> None:
    with pytest.raises(ValidationError, match="replaces"):
        GuardrailVerdict(outcome=GuardrailOutcome.REDACT)


def test_a_redaction_may_replace_content_with_nothing() -> None:
    assert GuardrailVerdict.redact("").content == ""


def test_a_block_without_a_reason_is_refused() -> None:
    with pytest.raises(ValidationError, match="why"):
        GuardrailVerdict(outcome=GuardrailOutcome.BLOCK)


def test_a_block_carries_its_reason_and_category() -> None:
    verdict = GuardrailVerdict.block("contains a card number", category="pii")
    assert (verdict.reason, verdict.category) == ("contains a card number", "pii")


# The pipeline


def test_an_empty_pipeline_allows_everything_unchanged() -> None:
    report = GuardrailPipeline().run("hello", GuardrailStage.INPUT)
    assert (report.outcome, report.content, report.verdicts) == (
        GuardrailOutcome.ALLOW,
        "hello",
        [],
    )


def test_guardrails_that_all_allow_leave_the_content_alone() -> None:
    report = GuardrailPipeline(allow("a"), allow("b")).run("hi", GuardrailStage.INPUT)
    assert report.outcome == GuardrailOutcome.ALLOW
    assert report.content == "hi"


def test_the_first_block_wins_and_nothing_after_it_runs() -> None:
    later = allow("later")
    report = GuardrailPipeline(block("first"), later).run("hi", GuardrailStage.INPUT)
    assert report.blocked
    assert later.seen == []


def test_a_block_after_an_allow_still_blocks() -> None:
    report = GuardrailPipeline(allow("a"), block("b")).run("hi", GuardrailStage.INPUT)
    assert report.blocked
    assert report.blocking_verdict is not None
    assert report.blocking_verdict.guardrail == "b"


def test_redactions_compose_in_the_order_the_guardrails_were_given() -> None:
    report = GuardrailPipeline(Upper(), Suffix("!")).run("hi", GuardrailStage.OUTPUT)
    assert report.content == "HI!"
    assert report.outcome == GuardrailOutcome.REDACT


def test_a_later_guardrail_sees_what_an_earlier_redaction_left() -> None:
    watcher = allow("watcher")
    GuardrailPipeline(Upper(), watcher).run("hi", GuardrailStage.OUTPUT)
    assert watcher.seen == ["HI"]


def test_a_block_after_a_redaction_blocks_on_the_redacted_content() -> None:
    blocker = block("late")
    report = GuardrailPipeline(Upper(), blocker).run("hi", GuardrailStage.OUTPUT)
    assert report.blocked
    assert blocker.seen == ["HI"]


def test_a_guardrail_is_only_consulted_at_the_stages_it_declares() -> None:
    tool_only = block("tool-only", stages=frozenset({GuardrailStage.TOOL_RESULT}))
    pipeline = GuardrailPipeline(tool_only)
    assert pipeline.run("hi", GuardrailStage.INPUT).outcome == GuardrailOutcome.ALLOW
    assert pipeline.run("hi", GuardrailStage.TOOL_RESULT).blocked
    assert tool_only.seen == ["hi"]


def test_a_pipeline_says_whether_anything_checks_a_stage() -> None:
    pipeline = GuardrailPipeline(allow(stages=frozenset({GuardrailStage.OUTPUT})))
    assert pipeline.applies_to(GuardrailStage.OUTPUT)
    assert not pipeline.applies_to(GuardrailStage.INPUT)


def test_every_verdict_names_the_guardrail_that_reached_it() -> None:
    report = GuardrailPipeline(allow("a"), Upper()).run("hi", GuardrailStage.INPUT)
    assert [v.guardrail for v in report.verdicts] == ["a", "upper"]


def test_two_guardrails_with_one_name_are_refused() -> None:
    with pytest.raises(ValueError, match="Duplicate guardrail names"):
        GuardrailPipeline(allow("same"), block("same"))


def test_the_report_records_the_stage_it_ran_at() -> None:
    report = GuardrailPipeline(allow()).run("hi", GuardrailStage.TOOL_RESULT)
    assert report.stage == GuardrailStage.TOOL_RESULT


def test_an_unblocked_report_has_no_blocking_verdict() -> None:
    report = GuardrailPipeline(allow()).run("hi", GuardrailStage.INPUT)
    assert report.blocking_verdict is None


# The async path


async def test_the_async_pipeline_applies_the_same_rules() -> None:
    later = allow("later")
    report = await GuardrailPipeline(Upper(), block("b"), later).arun(
        "hi", GuardrailStage.OUTPUT
    )
    assert report.blocked
    assert later.seen == []


async def test_the_async_pipeline_composes_redactions() -> None:
    report = await GuardrailPipeline(Upper(), Suffix("?")).arun(
        "hi", GuardrailStage.OUTPUT
    )
    assert report.content == "HI?"


# Enforcing


def test_enforce_returns_what_may_proceed() -> None:
    assert GuardrailPipeline(Upper()).enforce("hi", GuardrailStage.INPUT) == "HI"


def test_enforce_raises_on_a_block_carrying_the_report() -> None:
    with pytest.raises(GuardrailBlocked) as caught:
        GuardrailPipeline(block("b")).enforce("hi", GuardrailStage.INPUT)
    assert isinstance(caught.value.report, GuardrailReport)
    assert caught.value.report.blocking_verdict is not None
    assert caught.value.report.blocking_verdict.guardrail == "b"


async def test_async_enforce_raises_on_a_block() -> None:
    with pytest.raises(GuardrailBlocked):
        await GuardrailPipeline(block()).aenforce("hi", GuardrailStage.INPUT)


def test_a_block_reaches_a_caller_as_the_generic_content_policy_error() -> None:
    """The response names no guardrail: saying which check fired helps probe it."""
    with pytest.raises(GuardrailBlocked) as caught:
        GuardrailPipeline(block("pii")).enforce("hi", GuardrailStage.INPUT)
    assert caught.value.error_enum is GuardrailErrors.CONTENT_BLOCKED
    assert "pii" not in GuardrailErrors.CONTENT_BLOCKED.value["message"]


def test_the_content_policy_error_is_unprocessable_rather_than_a_server_fault() -> None:
    assert GuardrailErrors.CONTENT_BLOCKED.value["status_code"] == 422


# Reaching a caller


def test_a_block_raised_in_a_route_is_answered_as_a_content_policy_refusal(
    make_settings: Any,
) -> None:
    """Registered for `ErrorsException`, so the subclass is handled without wiring."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from base.tools.exception_handlers import AppExceptionHandler

    make_settings()
    app = FastAPI()
    AppExceptionHandler().register(app)

    @app.get("/ask")
    async def ask() -> dict[str, str]:
        GuardrailPipeline(block("pii")).enforce("hi", GuardrailStage.INPUT)
        return {"answer": "unreachable"}

    response = TestClient(app).get("/ask")
    assert response.status_code == 422
    assert response.json()["error"] == GuardrailErrors.CONTENT_BLOCKED.value["message"]
