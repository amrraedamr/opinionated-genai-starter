"""Spending a token budget around a model call.

Tokens are the budget that costs real money, and the only one whose size is
not known until after the work is done. So it is spent in two steps: ask
before the call whether anything is left, and record what was actually used
once the provider says. Reserving an estimate up front would either refuse
callers who would have fitted, or let a long reply overrun silently.

A streamed reply is no different here. The seam returns once the stream has
finished and the usage has arrived, so the spend happens at the same place.
"""

from base.models.definitions.errors import RateLimitErrors
from base.models.definitions.rate_limiting import RateLimitPolicy
from base.models.view_models.rate_limiting import RateDecision
from base.services.rate_limiting import AbstractRateLimiter
from base.services.rate_limiting.factory import build_rate_limiter
from base.tools.errors import ErrorsException
from base.tools.tenancy import TenantContext


class LLMBudgetExhausted(ErrorsException):
    """The caller has no tokens left this minute."""

    def __init__(self, decision: RateDecision) -> None:
        super().__init__(error_enum=RateLimitErrors.LLM_BUDGET_EXHAUSTED)
        self.decision = decision

    @property
    def retry_after_seconds(self) -> float:
        return self.decision.retry_after_seconds


class TokenBudget:
    """Asks before a model call, and records what it cost afterwards.

    Keyed by tenant when there is one. Without tenancy every caller shares one
    budget, which still bounds what a service can spend in a minute — it just
    cannot tell two customers apart, which is what tenancy is for.
    """

    def __init__(self, limiter: AbstractRateLimiter | None = None) -> None:
        self._limiter = limiter

    @property
    def limiter(self) -> AbstractRateLimiter:
        """Resolved on first use: settings are not registered at import."""
        if self._limiter is None:
            self._limiter = build_rate_limiter()
        return self._limiter

    @property
    def key(self) -> str:
        return AbstractRateLimiter.compose_key(tenant=TenantContext.get())

    async def acheck(self) -> None:
        """Refuse before the call when nothing is left.

        A zero cost asks without spending, so merely checking cannot itself
        exhaust the budget. When nothing is left, one more probe asks for a
        single token purely to learn the wait — it is certain to be refused,
        so it spends nothing, and a refusal with no retry is no use to a
        caller.
        """
        decision = await self.limiter.aspend(
            self.key, RateLimitPolicy.LLM_TOKENS, cost=0
        )
        if decision.allowed and decision.remaining > 0:
            return
        raise LLMBudgetExhausted(
            await self.limiter.aspend(self.key, RateLimitPolicy.LLM_TOKENS, cost=1)
        )

    def check(self) -> None:
        """Refuse before the call when nothing is left."""
        decision = self.limiter.spend(self.key, RateLimitPolicy.LLM_TOKENS, cost=0)
        if decision.allowed and decision.remaining > 0:
            return
        raise LLMBudgetExhausted(
            self.limiter.spend(self.key, RateLimitPolicy.LLM_TOKENS, cost=1)
        )

    async def aspend(self, tokens: int) -> None:
        """Record what the call actually used.

        Never raises. The work is done and the provider has been paid; failing
        here would lose the accounting and the reply with it. Going over simply
        means the next caller is refused.
        """
        if tokens > 0:
            await self.limiter.aspend(self.key, RateLimitPolicy.LLM_TOKENS, tokens)

    def spend(self, tokens: int) -> None:
        """Record what the call actually used. Never raises."""
        if tokens > 0:
            self.limiter.spend(self.key, RateLimitPolicy.LLM_TOKENS, tokens)


__all__ = ["LLMBudgetExhausted", "TokenBudget"]
