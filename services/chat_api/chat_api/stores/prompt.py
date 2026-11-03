from typing import ClassVar

from base.models.view_models.prompt import PromptTemplate
from base.services.common.meta import SingletonABC
from base.services.store.sql import SQLStore
from chat_api.models.orm.prompt import ActivePromptDTO


class PromptStore(SQLStore[ActivePromptDTO], metaclass=SingletonABC):
    """The prompt version currently active for each key."""

    protected_fields: ClassVar[frozenset[str]] = SQLStore.protected_fields | {"key"}
    search_fields: ClassVar[tuple[str, ...]] = ("key",)

    def __init__(self) -> None:
        super().__init__(ActivePromptDTO)

    def get_active_templates(self) -> dict[str, PromptTemplate]:
        return self._as_templates(self._many(self._active()))

    async def aget_active_templates(self) -> dict[str, PromptTemplate]:
        return self._as_templates(await self._amany(self._active()))

    @staticmethod
    def _as_templates(rows: list[ActivePromptDTO]) -> dict[str, PromptTemplate]:
        return {
            row.key: PromptTemplate.model_validate_json(row.prompt.template)
            for row in rows
        }
