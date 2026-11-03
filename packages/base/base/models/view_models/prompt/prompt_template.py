from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field, model_validator

from base.models.definitions.message import (
    MessageRole,
)
from base.models.view_models import BaseModel
from base.tools.prompt import (
    extract_template_variables,
    safe_format,
)

from .rendered_prompt import (
    RenderedPrompt,
    RenderedPromptInstruction,
    RenderedPromptItem,
    RenderedPromptMessage,
)


class BasePromptTemplateItem(BaseModel, ABC):
    content: str = ""

    def render_content(self, variables: dict[str, Any]) -> str:
        return self.content.format(**variables)

    def extract_template_variables(self) -> set[str]:
        return extract_template_variables(self.content)

    @abstractmethod
    def to_rendered_prompt_items(
        self,
        variables: dict[str, Any],
    ) -> list[RenderedPromptItem]:
        """Resolve this template item against the runtime variables.

        Most items render to exactly one :class:`RenderedPromptItem`
        (one system block, one user message, etc.), but some — notably
        :class:`MessagesPlaceholder` — fan out into a variable-length
        list. Returning a list everywhere lets the placeholder fit
        the same interface without a special case at the renderer.
        """
        raise NotImplementedError


class BaseMessageTemplate(BasePromptTemplateItem, ABC):
    role: MessageRole

    def to_rendered_prompt_items(
        self,
        variables: dict[str, Any],
    ) -> list[RenderedPromptItem]:
        rendered_content = self.render_content(variables)

        if self.role == MessageRole.SYSTEM:
            return [
                RenderedPromptInstruction(
                    role=MessageRole.SYSTEM,
                    content=rendered_content,
                )
            ]

        if self.role == MessageRole.DEVELOPER:
            return [
                RenderedPromptInstruction(
                    role=MessageRole.DEVELOPER,
                    content=rendered_content,
                )
            ]

        return [
            RenderedPromptMessage(
                role=self.role,
                content=rendered_content,
            )
        ]


class DeveloperMessageTemplate(BaseMessageTemplate):
    role: MessageRole = MessageRole.DEVELOPER


class SystemMessageTemplate(BaseMessageTemplate):
    role: MessageRole = MessageRole.SYSTEM


class UserMessageTemplate(BaseMessageTemplate):
    role: MessageRole = MessageRole.USER


class AssistantMessageTemplate(BaseMessageTemplate):
    role: MessageRole = MessageRole.ASSISTANT


class ToolMessageTemplate(BaseMessageTemplate):
    role: MessageRole = MessageRole.TOOL
    tool_call_id: str

    def extract_template_variables(self) -> set[str]:
        variables = super().extract_template_variables()
        variables.update(extract_template_variables(self.tool_call_id))
        return variables

    def to_rendered_prompt_items(
        self,
        variables: dict[str, Any],
    ) -> list[RenderedPromptItem]:
        return [
            RenderedPromptMessage(
                role=self.role,
                content=self.render_content(variables),
                tool_call_id=safe_format(self.tool_call_id, variables),
            )
        ]


class MessagesPlaceholder(BasePromptTemplateItem):
    """A slot in a message-based template that expands at render time
    into the list of messages stored under ``variable_name`` in the
    render kwargs.

    Mirrors LangChain's ``MessagesPlaceholder``. Use it to splice
    chat history (or any pre-built conversation thread) into a typed
    message template at a chosen position relative to the system
    instructions and the structured user content.

    The expected shape under ``variable_name`` is a list of items
    that each carry ``role`` + ``content`` (and optionally
    ``tool_call_id`` for tool messages). Pydantic ``StoredMessage``
    instances are accepted as-is; plain dicts work too. ``None`` and
    missing values resolve to an empty list when ``optional`` is
    ``True`` (the default).

    System / developer messages aren't supported inside the
    placeholder — chat history is conversation, not instructions.
    """

    variable_name: str
    optional: bool = True
    content: str = ""

    def render_content(self, variables: dict[str, Any]) -> str:
        return ""

    def extract_template_variables(self) -> set[str]:
        return {self.variable_name}

    def to_rendered_prompt_items(
        self,
        variables: dict[str, Any],
    ) -> list[RenderedPromptItem]:
        raw = variables.get(self.variable_name)
        if raw is None:
            if self.optional:
                return []
            raise ValueError(
                f"MessagesPlaceholder '{self.variable_name}' is required "
                f"but no value was provided."
            )

        items: list[RenderedPromptItem] = []
        for entry in raw:
            role, content, tool_call_id = self._coerce_message(entry)
            if role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
                raise ValueError(
                    f"MessagesPlaceholder '{self.variable_name}' received "
                    f"a {role.value} message; placeholders carry conversation "
                    f"messages, not instructions."
                )
            items.append(
                RenderedPromptMessage(
                    role=role,
                    content=content or "",
                    tool_call_id=tool_call_id,
                )
            )
        return items

    @staticmethod
    def _coerce_message(
        entry: Any,
    ) -> tuple[MessageRole, str | None, str | None]:
        if isinstance(entry, dict):
            role_value = entry.get("role")
            if isinstance(role_value, MessageRole):
                role = role_value
            else:
                role = MessageRole(role_value)
            return role, entry.get("content"), entry.get("tool_call_id")

        # Pydantic / dataclass / object with .role / .content attributes.
        role_value = entry.role
        role = (
            role_value
            if isinstance(role_value, MessageRole)
            else MessageRole(role_value)
        )
        return (
            role,
            getattr(entry, "content", None),
            getattr(entry, "tool_call_id", None),
        )


PromptMessageTemplate = (
    DeveloperMessageTemplate
    | SystemMessageTemplate
    | UserMessageTemplate
    | AssistantMessageTemplate
    | ToolMessageTemplate
    | MessagesPlaceholder
)


class PromptTemplate(BaseModel):
    template: str | None = None
    messages: list[PromptMessageTemplate] = Field(default_factory=list)
    input_variables: list[str] = Field(default_factory=list)
    partial_variables: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_input_variables(self) -> "PromptTemplate":
        if self.template and self.messages:
            raise ValueError("Provide either 'template' or 'messages', not both.")

        if not self.template and not self.messages:
            raise ValueError("Either 'template' or 'messages' must be provided.")

        detected_variables: set[str] = set()

        if self.template:
            detected_variables.update(extract_template_variables(self.template))

        for message in self.messages:
            detected_variables.update(message.extract_template_variables())

        unknown_partials = set(self.partial_variables) - detected_variables
        if unknown_partials:
            raise ValueError(
                f"partial_variables contains keys not referenced by the template: "
                f"{sorted(unknown_partials)}"
            )

        detected_variables -= set(self.partial_variables)

        if self.input_variables:
            declared = set(self.input_variables)
            if declared != detected_variables:
                raise ValueError(
                    f"input_variables mismatch. "
                    f"Declared={sorted(declared)}, Detected={sorted(detected_variables)}"
                )

        object.__setattr__(self, "input_variables", sorted(detected_variables))
        return self

    @classmethod
    def from_messages(
        cls,
        messages: list[PromptMessageTemplate],
    ) -> "PromptTemplate":
        return cls(messages=messages)

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        return cls(template=template)

    def partial(self, **kwargs: Any) -> "PromptTemplate":
        unknown = set(kwargs) - set(self.input_variables) - set(self.partial_variables)
        if unknown:
            raise ValueError(
                f"partial() received keys not in input_variables: {sorted(unknown)}"
            )
        prompt_copy = self.model_copy(deep=True)
        return PromptTemplate(
            template=prompt_copy.template,
            messages=prompt_copy.messages,
            partial_variables={**prompt_copy.partial_variables, **kwargs},
        )

    def _merge_and_validate(self, runtime_variables: dict[str, Any]) -> dict[str, Any]:
        variables = {**self.partial_variables, **runtime_variables}

        missing = [var for var in self.input_variables if var not in variables]
        if missing:
            raise ValueError(f"Missing required input variables: {missing}")

        return variables

    def format(self, **kwargs: Any) -> str:
        """Render a string-only template."""
        variables = self._merge_and_validate(kwargs)

        if not self.template:
            raise ValueError("This PromptTemplate is message-based, not string-based.")

        return self.template.format(**variables)

    def render_prompt(self, **kwargs: Any) -> RenderedPrompt:
        variables = self._merge_and_validate(kwargs)

        if not self.messages:
            assert self.template is not None  # guaranteed by validator
            return RenderedPrompt(
                variables=dict(variables),
                messages=[
                    RenderedPromptMessage(
                        role=MessageRole.USER,
                        content=self.template.format(**variables),
                    )
                ],
            )

        instructions: list[RenderedPromptInstruction] = []
        messages: list[RenderedPromptMessage] = []

        for template_item in self.messages:
            for rendered_item in template_item.to_rendered_prompt_items(variables):
                if isinstance(rendered_item, RenderedPromptInstruction):
                    instructions.append(rendered_item)
                else:
                    messages.append(rendered_item)

        return RenderedPrompt(
            variables=dict(variables),
            instructions=instructions,
            messages=messages,
        )
