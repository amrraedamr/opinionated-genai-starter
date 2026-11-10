"""Spending a token budget around a model call."""

from base.models.definitions.errors import RateLimitErrors
from base.models.definitions.rate_limiting import RateLimitPolicy
from base.models.view_models.rate_limiting import RateDecision
from base.services.rate_limiting import AbstractRateLimiter
from base.services.rate_limiting.factory import build_rate_limiter
from base.tools.errors import ErrorsException
from base.tools.tenancy import CallerContext, TenantContext


class LLMBudgetExhausted(ErrorsException):
    """The caller has no tokens left this minute."""

    def __init__(self, decision: RateDecision) -> None:
        super().__init__(error_enum=RateLimitErrors.LLM_BUDGET_EXHAUSTED)
        self.decision = decision

    @property
    def retry_after_seconds(self) -> float:
        return self.decision.retry_after_seconds


class TokenBudget:
    """Asks before a model call, and records what it cost afterwards; per user within a tenant."""

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
        return AbstractRateLimiter.compose_key(
            tenant=TenantContext.get(), user=CallerContext.get()
        )

    async def acheck(self) -> None:
        """Refuse before the call when nothing is left, asynchronously."""
        decision = await self.limiter.aspend(
            self.key, RateLimitPolicy.LLM_TOKENS, cost=0
        )
        if decision.remaining == 0:
            raise LLMBudgetExhausted(decision)

    def check(self) -> None:
        """Refuse before the call when nothing is left."""
        decision = self.limiter.spend(self.key, RateLimitPolicy.LLM_TOKENS, cost=0)
        if decision.remaining == 0:
            raise LLMBudgetExhausted(decision)

    async def aspend(self, tokens: int) -> None:
        """Record what the call used, even past the limit, asynchronously."""
        if tokens > 0:
            await self.limiter.acharge(self.key, RateLimitPolicy.LLM_TOKENS, tokens)

    def spend(self, tokens: int) -> None:
        """Record what the call used, even past the limit."""
        if tokens > 0:
            self.limiter.charge(self.key, RateLimitPolicy.LLM_TOKENS, tokens)


__all__ = ["LLMBudgetExhausted", "TokenBudget"]
