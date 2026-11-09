"""Token budgets at the LLM boundary, for every provider at once.

Tokens are the budget that costs real money and the only one whose size is
unknown until the work is done, so it is spent in two steps: ask before,
record after. These pin both halves, and that going over is not retroactive —
the call that exceeded the budget still returns, and the next one is refused.

The wiring lives in `AbstractLLMService`'s final wrappers, so a stand-in
subclass exercises exactly the path every real adapter takes.
"""

import uuid
from typing import Any

import pytest

from base.models.definitions.errors import RateLimitErrors
from base.models.definitions.message import MessageRole
from base.models.definitions.rate_limiting import RateLimitPolicy
from base.models.definitions.tenancy import TenantId
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import ConversationId, StoredMessage
from base.models.view_models.llm_result import (
    LLMResult,
    LLMUsageCounts,
    ParsedResult,
)
from base.services.configuration.mixins import (
    RateLimitSettingsMixin,
    TenancySettingsMixin,
)
from base.services.llm import AbstractLLMService
from base.services.rate_limiting.llm_budget import LLMBudgetExhausted, TokenBudget
from base.services.rate_limiting.memory import InMemoryRateLimiter
from base.tools.tenancy import TenantContext

CONVERSATION = ConversationId(uuid.UUID(int=11))


class Answer(BaseModel):
    city: str = "Doha"


class ScriptedLLM(AbstractLLMService):
    """A provider that reports whatever usage the test chose."""

    def __init__(self, tokens: int = 10) -> None:
        self.tokens = tokens
        self.calls = 0

    @property
    def model(self) -> str:
        return "scripted"

    def get_service_name(self) -> str:
        return "scripted"

    def _apply_cache_hint(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def _usage(self) -> LLMUsageCounts:
        return LLMUsageCounts(input_tokens=self.tokens, output_tokens=0)

    def _complete(self, *args: Any, **kwargs: Any) -> LLMResult:
        self.calls += 1
        return LLMResult(content="ok", usage=self._usage())

    async def _acomplete(self, *args: Any, **kwargs: Any) -> LLMResult:
        return self._complete()

    def _parse(self, schema: Any, *args: Any, **kwargs: Any) -> ParsedResult[Any]:
        self.calls += 1
        return ParsedResult(parsed=schema(), usage=self._usage())

    async def _aparse(
        self, schema: Any, *args: Any, **kwargs: Any
    ) -> ParsedResult[Any]:
        return self._parse(schema)


def message(content: str = "hello") -> StoredMessage:
    return StoredMessage(
        conversation_id=CONVERSATION,
        sequence=1,
        role=MessageRole.USER,
        content=content,
    )


@pytest.fixture
def budgeted(make_settings: Any) -> Any:
    """A hundred tokens a minute, counted in this process."""

    def _build(tokens_per_minute: int = 100, **extra: Any) -> ScriptedLLM:
        make_settings(
            RateLimitSettingsMixin,
            TenancySettingsMixin,
            rate_limit_llm_tokens_per_minute=tokens_per_minute,
            **extra,
        )
        return ScriptedLLM()

    return _build


async def test_a_call_inside_the_budget_goes_through(budgeted: Any) -> None:
    service = budgeted()
    result = await service.acomplete(messages=[message()])
    assert result.content == "ok"


async def test_the_tokens_a_call_used_are_recorded(budgeted: Any) -> None:
    """Recorded after the fact, because the size is unknown until then."""
    service = budgeted(tokens_per_minute=100)
    service.tokens = 40
    await service.acomplete(messages=[message()])

    limiter = InMemoryRateLimiter()
    remaining = await limiter.aspend(
        TokenBudget().key, RateLimitPolicy.LLM_TOKENS, cost=0
    )
    assert remaining.remaining == 60


async def test_the_next_call_is_refused_once_the_budget_is_spent(
    budgeted: Any,
) -> None:
    service = budgeted(tokens_per_minute=50)
    service.tokens = 50
    await service.acomplete(messages=[message()])

    with pytest.raises(LLMBudgetExhausted):
        await service.acomplete(messages=[message()])


async def test_the_call_that_went_over_still_returns(budgeted: Any) -> None:
    """The provider has already been paid; losing the reply helps nobody."""
    service = budgeted(tokens_per_minute=10)
    service.tokens = 500
    result = await service.acomplete(messages=[message()])
    assert result.content == "ok"


async def test_a_refusal_never_reaches_the_provider(budgeted: Any) -> None:
    service = budgeted(tokens_per_minute=10)
    service.tokens = 10
    await service.acomplete(messages=[message()])
    before = service.calls

    with pytest.raises(LLMBudgetExhausted):
        await service.acomplete(messages=[message()])
    assert service.calls == before


async def test_the_refusal_is_a_domain_error(budgeted: Any) -> None:
    """Not a raw limiter type: a caller catches this family already."""
    service = budgeted(tokens_per_minute=5)
    service.tokens = 5
    await service.acomplete(messages=[message()])

    with pytest.raises(LLMBudgetExhausted) as refused:
        await service.acomplete(messages=[message()])

    assert refused.value.error_enum is RateLimitErrors.LLM_BUDGET_EXHAUSTED
    assert refused.value.retry_after_seconds > 0


def test_the_sync_path_is_budgeted_too(budgeted: Any) -> None:
    service = budgeted(tokens_per_minute=10)
    service.tokens = 10
    service.complete(messages=[message()])

    with pytest.raises(LLMBudgetExhausted):
        service.complete(messages=[message()])


async def test_a_parsed_call_is_budgeted_too(budgeted: Any) -> None:
    """`aparse` is a separate wrapper and would otherwise spend nothing."""
    service = budgeted(tokens_per_minute=10)
    service.tokens = 10
    await service.aparse(Answer, messages=[message()])

    with pytest.raises(LLMBudgetExhausted):
        await service.aparse(Answer, messages=[message()])


def test_a_parsed_sync_call_is_budgeted_too(budgeted: Any) -> None:
    service = budgeted(tokens_per_minute=10)
    service.tokens = 10
    service.parse(Answer, messages=[message()])

    with pytest.raises(LLMBudgetExhausted):
        service.parse(Answer, messages=[message()])


async def test_a_reply_with_no_usage_spends_nothing(budgeted: Any) -> None:
    """A provider that reports nothing must not be charged a guess."""
    service = budgeted(tokens_per_minute=10)

    async def no_usage(*args: Any, **kwargs: Any) -> LLMResult:
        return LLMResult(content="ok")

    service._acomplete = no_usage  # type: ignore[method-assign]
    for _ in range(5):
        await service.acomplete(messages=[message()])


async def test_tenants_do_not_share_a_token_budget(budgeted: Any) -> None:
    """The point of keying by tenant: one customer cannot spend another's."""
    service = budgeted(tokens_per_minute=20, tenancy_enabled=True)
    service.tokens = 20

    with TenantContext.scope(TenantId("acme")):
        await service.acomplete(messages=[message()])
        with pytest.raises(LLMBudgetExhausted):
            await service.acomplete(messages=[message()])

    with TenantContext.scope(TenantId("globex")):
        assert (await service.acomplete(messages=[message()])).content == "ok"


async def test_without_a_tenant_one_budget_bounds_the_service(
    budgeted: Any,
) -> None:
    """Still useful: it caps what the service spends in a minute."""
    service = budgeted(tokens_per_minute=20)
    service.tokens = 20
    await service.acomplete(messages=[message()])
    with pytest.raises(LLMBudgetExhausted):
        await service.acomplete(messages=[message()])


async def test_checking_does_not_itself_spend(budgeted: Any) -> None:
    """A zero cost asks what is left; asking must not consume it."""
    budgeted(tokens_per_minute=100)
    budget = TokenBudget()
    for _ in range(10):
        await budget.acheck()

    decision = await budget.limiter.aspend(
        budget.key, RateLimitPolicy.LLM_TOKENS, cost=0
    )
    assert decision.remaining == 100
