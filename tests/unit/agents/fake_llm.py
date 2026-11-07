"""A stand-in LLM service for agent tests, and the agent factory they share.

`AbstractAgent` builds its provider from `LLM_SERVICE()`, so the fake has to be
a class the agent can construct itself. What matters in these tests is the
request an agent assembles, so every keyword is kept rather than discarded.

Agent classes are built per test: `AbstractAgent` is a `SingletonABC`, so one
instance exists per type for the life of the process and a class reused across
tests would answer with the previous test's collaborators.
"""

from __future__ import annotations

import itertools
from typing import Any

from base.models.view_models import BaseModel
from base.models.view_models.llm_result import LLMResult, ParsedResult
from base.services.agents import AbstractAgent
from base.services.common.meta import SingletonABC

_names = itertools.count()


class RecordingLLM:
    """Answers like an `AbstractLLMService`, keeping every kwarg it is handed."""

    def __init__(self) -> None:
        self.complete_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []
        self.reply: LLMResult = LLMResult(content="ok")

    async def acomplete(self, **kwargs: Any) -> LLMResult:
        self.complete_calls.append(kwargs)
        return self.reply

    async def aparse[T: BaseModel](
        self, schema: type[T], **kwargs: Any
    ) -> ParsedResult[T]:
        self.parse_calls.append({"schema": schema, **kwargs})
        return ParsedResult(parsed=schema())

    @property
    def last_complete(self) -> dict[str, Any]:
        assert self.complete_calls, "no completion was requested"
        return self.complete_calls[-1]

    @property
    def last_parse(self) -> dict[str, Any]:
        assert self.parse_calls, "no parse was requested"
        return self.parse_calls[-1]


class Answer(BaseModel):
    """The smallest schema a parsed call can be asked for."""

    text: str = ""


def agent_class(**class_vars: Any) -> type[AbstractAgent]:
    """A uniquely named concrete `AbstractAgent` subclass with these ClassVars."""

    async def _execute(self: AbstractAgent) -> None:
        return None

    namespace: dict[str, Any] = {"_execute": _execute, **class_vars}
    namespace.setdefault("LLM_SERVICE", RecordingLLM)
    return SingletonABC(f"Agent{next(_names)}", (AbstractAgent,), namespace)


def build_agent(**class_vars: Any) -> AbstractAgent:
    """One instance of a freshly minted agent class."""
    return agent_class(**class_vars)()


def llm_of(agent: AbstractAgent) -> RecordingLLM:
    """The fake the agent built for itself."""
    service = agent._llm_service
    assert isinstance(service, RecordingLLM)
    return service
