from typing import Any

from base.models.view_models.prompt import (
    Prompt,
    PromptTemplate,
    RenderedPrompt,
)
from base.services.common.meta import Singleton
from base.services.prompt.loader import PromptLoader


class PromptService(metaclass=Singleton):
    def __init__(self) -> None:
        self._loader: PromptLoader = PromptLoader()

    def get(self, key: str) -> Prompt:
        return self._loader.get(key)

    def get_raw(self, key: str) -> PromptTemplate:
        return self.get(key).template

    def format(self, key: str, **variables: Any) -> str:
        return self.get(key).template.format(**variables)

    def render(
        self,
        key: str,
        **variables: Any,
    ) -> RenderedPrompt:
        return self.get(key).render(**variables)

    def list_keys(self) -> list[str]:
        return self._loader.list_keys()
