from typing import Protocol

from base.models.view_models.prompt import PromptTemplate


class PromptTemplateStore(Protocol):
    """Whatever holds a service's prompt templates outside its code.

    The registry asks for templates by key and never learns which database
    answered, so a service is free to keep them wherever it keeps its data.
    """

    async def aget_active_templates(self) -> dict[str, PromptTemplate]: ...
