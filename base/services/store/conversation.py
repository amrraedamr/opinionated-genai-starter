from typing import Any, ClassVar
from uuid import UUID

from sqlalchemy import select

from base.models.orm.chat_history_message import ChatHistoryMessageDTO
from base.models.orm.conversation import ConversationDTO
from base.models.view_models.conversation import (
    CreateConversationRequest,
    EditConversationRequest,
)
from base.models.view_models.database import FilteredPaginationParams, PaginatedResult
from base.services.common.meta import SingletonABC
from base.services.store.sql import SQLStore
from base.tools.database import SQLResultFormats


class ConversationStore(SQLStore[ConversationDTO], metaclass=SingletonABC):
    """Conversations, always scoped to the user who owns them.

    The `*_for_user` methods are the ones routers should call: they make
    ownership part of the query rather than a check a caller might forget.
    """

    protected_fields: ClassVar[frozenset[str]] = SQLStore.protected_fields | {"user_id"}
    search_fields: ClassVar[tuple[str, ...]] = ("title",)

    def __init__(self) -> None:
        super().__init__(ConversationDTO)

    # Messages die with their conversation

    def _message_query(self, conversation_id: UUID) -> Any:
        return select(ChatHistoryMessageDTO).where(
            ChatHistoryMessageDTO.conversation_id == conversation_id,
            ChatHistoryMessageDTO.deleted_at.is_(None),
        )

    def _cascade(self, entity: ConversationDTO) -> list[Any]:
        return self._sql_service.query(
            self._message_query(entity.id), SQLResultFormats.all_scalar_list
        )

    async def _acascade(self, entity: ConversationDTO) -> list[Any]:
        return await self._sql_service.aquery(
            self._message_query(entity.id), SQLResultFormats.all_scalar_list
        )

    # Helpers

    def _apply_edit_request(
        self, existing: ConversationDTO, request: EditConversationRequest
    ) -> ConversationDTO:
        update_data = request.serialize(
            exclude_unset=True, exclude_none=True, by_alias=False
        )
        if "metadata" in update_data:
            update_data["meta"] = update_data.pop("metadata")

        existing.update_from_dict(self._sanitize_update_data(update_data))
        return existing

    @staticmethod
    def _build_conversation_entity(
        user_id: UUID, request: CreateConversationRequest
    ) -> ConversationDTO:
        return ConversationDTO(
            user_id=user_id,
            title=request.title,
            meta=request.metadata,
        )

    # Get

    def get_for_user(
        self, user_id: UUID, conversation_id: UUID
    ) -> ConversationDTO | None:
        return self._one(
            self._active(
                ConversationDTO.user_id == user_id,
                ConversationDTO.id == conversation_id,
            )
        )

    async def aget_for_user(
        self, user_id: UUID, conversation_id: UUID
    ) -> ConversationDTO | None:
        return await self._aone(
            self._active(
                ConversationDTO.user_id == user_id,
                ConversationDTO.id == conversation_id,
            )
        )

    def get_many_for_user(
        self, user_id: UUID, conversation_ids: list[UUID]
    ) -> list[ConversationDTO]:
        return self._many(
            self._active(
                ConversationDTO.user_id == user_id,
                ConversationDTO.id.in_(conversation_ids),
            )
        )

    async def aget_many_for_user(
        self, user_id: UUID, conversation_ids: list[UUID]
    ) -> list[ConversationDTO]:
        return await self._amany(
            self._active(
                ConversationDTO.user_id == user_id,
                ConversationDTO.id.in_(conversation_ids),
            )
        )

    def get_all_for_user(self, user_id: UUID) -> list[ConversationDTO]:
        return self._many(self._active(ConversationDTO.user_id == user_id))

    async def aget_all_for_user(self, user_id: UUID) -> list[ConversationDTO]:
        return await self._amany(self._active(ConversationDTO.user_id == user_id))

    def get_all_paged_for_user(
        self, user_id: UUID, filter_page_params: FilteredPaginationParams
    ) -> PaginatedResult[ConversationDTO]:
        return self._sql_service.paginate(
            self._filtered(filter_page_params, ConversationDTO.user_id == user_id),
            filter_page_params,
        )

    async def aget_all_paged_for_user(
        self, user_id: UUID, filter_page_params: FilteredPaginationParams
    ) -> PaginatedResult[ConversationDTO]:
        return await self._sql_service.apaginate(
            self._filtered(filter_page_params, ConversationDTO.user_id == user_id),
            filter_page_params,
        )

    # Add

    def add_for_user(self, user_id: UUID, entity: ConversationDTO) -> ConversationDTO:
        entity.user_id = user_id
        return self._sql_service.upsert_row(entity)

    async def aadd_for_user(
        self, user_id: UUID, entity: ConversationDTO
    ) -> ConversationDTO:
        entity.user_id = user_id
        return await self._sql_service.aupsert_row(entity)

    def add_many_for_user(
        self, user_id: UUID, entities: list[ConversationDTO]
    ) -> list[ConversationDTO]:
        for entity in entities:
            entity.user_id = user_id
        return self._sql_service.upsert_rows(entities)

    async def aadd_many_for_user(
        self, user_id: UUID, entities: list[ConversationDTO]
    ) -> list[ConversationDTO]:
        for entity in entities:
            entity.user_id = user_id
        return await self._sql_service.aupsert_rows(entities)

    def create_for_user(
        self, user_id: UUID, request: CreateConversationRequest
    ) -> ConversationDTO:
        return self._sql_service.upsert_row(
            self._build_conversation_entity(user_id, request)
        )

    async def acreate_for_user(
        self, user_id: UUID, request: CreateConversationRequest
    ) -> ConversationDTO:
        return await self._sql_service.aupsert_row(
            self._build_conversation_entity(user_id, request)
        )

    # Edit

    def edit_for_user(
        self, user_id: UUID, value: Any, request: EditConversationRequest
    ) -> ConversationDTO | None:
        existing = self.get_for_user(user_id, value)
        if existing is None:
            return None
        return self._sql_service.upsert_row(
            self._apply_edit_request(existing, request)
        )

    async def aedit_for_user(
        self, user_id: UUID, value: Any, request: EditConversationRequest
    ) -> ConversationDTO | None:
        existing = await self.aget_for_user(user_id, value)
        if existing is None:
            return None
        return await self._sql_service.aupsert_row(
            self._apply_edit_request(existing, request)
        )

    # Delete

    def delete_for_user(self, user_id: UUID, value: Any) -> bool:
        existing = self.get_for_user(user_id, value)
        return False if existing is None else self._soft_delete(existing)

    async def adelete_for_user(self, user_id: UUID, value: Any) -> bool:
        existing = await self.aget_for_user(user_id, value)
        return False if existing is None else await self._asoft_delete(existing)
