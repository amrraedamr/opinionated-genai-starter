from typing import Any, Literal, Self

from pydantic import Field, model_validator

from base.models.definitions.message import MessageRole
from base.models.view_models import BaseModel


class RenderedPromptInstruction(BaseModel):
    """A rendered system/developer instruction block."""

    role: Literal[MessageRole.SYSTEM, MessageRole.DEVELOPER]
    content: str


class RenderedPromptMessage(BaseModel):
    """A rendered prompt message that participates in conversation history."""

    role: MessageRole
    content: str
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
            raise ValueError(
                "RenderedPromptMessage cannot use 'system' or 'developer' roles."
            )

        if self.role == MessageRole.TOOL and self.tool_call_id is None:
            raise ValueError("Tool messages require tool_call_id.")

        if self.role != MessageRole.TOOL and self.tool_call_id is not None:
            raise ValueError("Only tool messages may define tool_call_id.")

        return self


RenderedPromptItem = RenderedPromptInstruction | RenderedPromptMessage


class RenderedPrompt(BaseModel):
    """Canonical rendered prompt output used by orchestration layers."""

    key: str | None = None
    name: str | None = None
    description: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    instructions: list[RenderedPromptInstruction] = Field(default_factory=list)
    messages: list[RenderedPromptMessage] = Field(default_factory=list)
