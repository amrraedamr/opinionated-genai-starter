"""Checking a reply that is being sent while it is still being written.

A streamed answer has already left by the time the last token arrives, so the
usual "check the output" does not apply. The policy here is one policy, written
once, because websockets will need exactly the same thing:

* Redact as it goes, holding a short tail. A card number split across two
  chunks is invisible to each of them on its own, so the end of each chunk is
  held back until the next one arrives.
* Check the whole assembled reply at the end. Some things — a refusal, a
  moderation category — can only be judged on the complete text.
* On a block at the end, emit a retraction rather than pretending it did not
  happen. The caller already has the text; the honest move is to say so.
"""

from base.models.definitions.guardrails import GuardrailOutcome, GuardrailStage
from base.models.view_models.guardrails import GuardrailReport
from base.models.view_models.streams import StreamError
from base.services.guardrails.pipeline import GuardrailPipeline

RETRACTION_CODE = 422
RETRACTION_TEXT = "The reply was withdrawn by a content policy."


class GuardedStream:
    """Redacts chunks on the way out, and judges the whole at the end."""

    def __init__(self, guardrails: GuardrailPipeline, tail: int = 32) -> None:
        self._guardrails = guardrails
        self._tail = max(tail, 0)
        self._held = ""
        self._assembled: list[str] = []

    @property
    def assembled(self) -> str:
        """Everything emitted so far, as the caller has received it."""
        return "".join(self._assembled)

    @property
    def active(self) -> bool:
        """Whether anything would actually run, so a caller can skip the work."""
        return self._guardrails.applies_to(GuardrailStage.OUTPUT)

    async def apush(self, chunk: str) -> str:
        """What may be emitted now, given what is still being held.

        The tail is withheld rather than emitted and corrected later: a caller
        that has already rendered a card number cannot un-see it.
        """
        if not self.active or not chunk:
            self._assembled.append(chunk)
            return chunk

        pending = self._held + chunk
        emit, self._held = self._split(pending)
        if not emit:
            return ""

        report = await self._guardrails.arun(emit, GuardrailStage.OUTPUT)
        if report.outcome == GuardrailOutcome.BLOCK:
            self._held = ""
            return ""
        self._assembled.append(report.content)
        return report.content

    async def aflush(self) -> str:
        """The held tail, checked, now that nothing more is coming."""
        if not self._held:
            return ""
        remainder, self._held = self._held, ""
        if not self.active:
            self._assembled.append(remainder)
            return remainder

        report = await self._guardrails.arun(remainder, GuardrailStage.OUTPUT)
        if report.outcome == GuardrailOutcome.BLOCK:
            return ""
        self._assembled.append(report.content)
        return report.content

    async def averdict(self) -> GuardrailReport | None:
        """The whole reply, judged. `None` when nothing objected.

        Redactions are ignored here: they were already applied chunk by chunk,
        and re-reporting them would retract a reply that was only cleaned.
        """
        if not self.active:
            return None
        report = await self._guardrails.arun(self.assembled, GuardrailStage.OUTPUT)
        return report if report.outcome == GuardrailOutcome.BLOCK else None

    @staticmethod
    def retraction() -> StreamError:
        """Says the reply is withdrawn, without saying which rule caught it."""
        return StreamError(error_code=RETRACTION_CODE, text=RETRACTION_TEXT)

    def _split(self, pending: str) -> tuple[str, str]:
        """Everything but the last `tail` characters, which stay held."""
        if len(pending) <= self._tail:
            return "", pending
        return pending[: -self._tail], pending[-self._tail :]


__all__ = ["RETRACTION_CODE", "RETRACTION_TEXT", "GuardedStream"]
