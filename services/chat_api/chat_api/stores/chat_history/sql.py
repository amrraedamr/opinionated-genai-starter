from sqlalchemy import Select, delete, func, select
from sqlalchemy.exc import IntegrityError

from base.models.view_models.chat_history import (
    ConversationId,
    StoredMessage,
    StoredToolCall,
)
from base.services.common.meta import SingletonABC
from base.services.sql.service import SQLService
from base.services.store.chat_history import ChatHistoryStore
from base.tools.database import SQLResultFormats
from chat_api.models.orm.chat_history_message import ChatHistoryMessageDTO


class SqlChatHistoryStore(ChatHistoryStore, metaclass=SingletonABC):
    """SQLAlchemy-backed ChatHistoryStore. One row per message.

    Concurrency: a unique constraint on (conversation_id, sequence) enforces
    correctness. The store does not retry on collision — it lets IntegrityError
    propagate. Upstream (ChatHistoryService) serializes anext_sequence and
    aappend per conversation so collisions don't reach the database.

    IDs are assigned by the database via ChatHistoryMessageDTO's uuid default.
    The service constructs StoredMessage drafts with id=None and the store
    populates id on return from aappend / areplace_range.
    """

    def __init__(self) -> None:
        self._sql_service: SQLService = SQLService()
        self._base_select_query: Select[tuple[ChatHistoryMessageDTO]] = select(
            ChatHistoryMessageDTO
        )

    def _duplicate_error_types(self) -> tuple[type[BaseException], ...]:
        return (IntegrityError,)

    # Mapping

    @staticmethod
    def _to_dto(message: StoredMessage) -> ChatHistoryMessageDTO:
        return ChatHistoryMessageDTO(
            conversation_id=message.conversation_id,
            sequence=message.sequence,
            role=message.role,
            content=message.content,
            tool_calls=(
                [tc.model_dump() for tc in message.tool_calls]
                if message.tool_calls is not None
                else None
            ),
            tool_call_id=message.tool_call_id,
            token_count=message.token_count,
            meta=message.metadata,
        )

    @staticmethod
    def _from_dto(dto: ChatHistoryMessageDTO) -> StoredMessage:
        return StoredMessage(
            id=dto.id,
            conversation_id=ConversationId(dto.conversation_id),
            sequence=dto.sequence,
            role=dto.role,
            content=dto.content,
            tool_calls=(
                [StoredToolCall.model_validate(tc) for tc in dto.tool_calls]
                if dto.tool_calls is not None
                else None
            ),
            tool_call_id=dto.tool_call_id,
            token_count=dto.token_count,
            metadata=dto.meta or {},
        )

    # Append

    def _append(
        self,
        parent_id: ConversationId,
        items: list[StoredMessage],
    ) -> list[StoredMessage]:
        dtos = [self._to_dto(m) for m in items]
        persisted = self._sql_service.upsert_rows(dtos)
        return [self._from_dto(d) for d in persisted]

    async def _aappend(
        self,
        parent_id: ConversationId,
        items: list[StoredMessage],
    ) -> list[StoredMessage]:
        dtos = [self._to_dto(m) for m in items]
        persisted = await self._sql_service.aupsert_rows(dtos)
        return [self._from_dto(d) for d in persisted]

    # Get

    def _get(
        self,
        parent_id: ConversationId,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        if limit is None:
            query = self._base_select_query.where(
                ChatHistoryMessageDTO.conversation_id == parent_id
            ).order_by(ChatHistoryMessageDTO.sequence.asc())
            dtos = self._sql_service.query(query, SQLResultFormats.all_scalar_list)
        else:
            query = (
                self._base_select_query.where(
                    ChatHistoryMessageDTO.conversation_id == parent_id
                )
                .order_by(ChatHistoryMessageDTO.sequence.desc())
                .limit(limit)
            )
            dtos = list(
                reversed(
                    self._sql_service.query(query, SQLResultFormats.all_scalar_list)
                )
            )
        return [self._from_dto(d) for d in dtos]

    async def _aget(
        self,
        parent_id: ConversationId,
        *,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        if limit is None:
            query = self._base_select_query.where(
                ChatHistoryMessageDTO.conversation_id == parent_id
            ).order_by(ChatHistoryMessageDTO.sequence.asc())
            dtos = await self._sql_service.aquery(
                query, SQLResultFormats.all_scalar_list
            )
        else:
            query = (
                self._base_select_query.where(
                    ChatHistoryMessageDTO.conversation_id == parent_id
                )
                .order_by(ChatHistoryMessageDTO.sequence.desc())
                .limit(limit)
            )
            dtos = list(
                reversed(
                    await self._sql_service.aquery(
                        query, SQLResultFormats.all_scalar_list
                    )
                )
            )
        return [self._from_dto(d) for d in dtos]

    # Count

    def _count(self, parent_id: ConversationId) -> int:
        query = select(func.count(ChatHistoryMessageDTO.id)).where(
            ChatHistoryMessageDTO.conversation_id == parent_id
        )
        result = self._sql_service.query(query, SQLResultFormats.one_or_none_scalar)
        return result or 0

    async def _acount(self, parent_id: ConversationId) -> int:
        query = select(func.count(ChatHistoryMessageDTO.id)).where(
            ChatHistoryMessageDTO.conversation_id == parent_id
        )
        result = await self._sql_service.aquery(
            query, SQLResultFormats.one_or_none_scalar
        )
        return result or 0

    # Next sequence

    def _next_sequence(self, parent_id: ConversationId) -> int:
        query = select(func.max(ChatHistoryMessageDTO.sequence)).where(
            ChatHistoryMessageDTO.conversation_id == parent_id
        )
        current = self._sql_service.query(query, SQLResultFormats.one_or_none_scalar)
        return (current or 0) + 1

    async def _anext_sequence(self, parent_id: ConversationId) -> int:
        query = select(func.max(ChatHistoryMessageDTO.sequence)).where(
            ChatHistoryMessageDTO.conversation_id == parent_id
        )
        current = await self._sql_service.aquery(
            query, SQLResultFormats.one_or_none_scalar
        )
        return (current or 0) + 1

    # Clear

    def _clear(self, parent_id: ConversationId) -> None:
        stmt = delete(ChatHistoryMessageDTO).where(
            ChatHistoryMessageDTO.conversation_id == parent_id
        )
        self._sql_service.delete_many(stmt)

    async def _aclear(self, parent_id: ConversationId) -> None:
        stmt = delete(ChatHistoryMessageDTO).where(
            ChatHistoryMessageDTO.conversation_id == parent_id
        )
        await self._sql_service.adelete_many(stmt)

    # Replace range

    def _replace_range(
        self,
        parent_id: ConversationId,
        *,
        up_to_sequence: int,
        replacements: list[StoredMessage],
    ) -> list[StoredMessage]:
        delete_stmt = delete(ChatHistoryMessageDTO).where(
            ChatHistoryMessageDTO.conversation_id == parent_id,
            ChatHistoryMessageDTO.sequence <= up_to_sequence,
        )
        replacement_dtos = [self._to_dto(m) for m in replacements]
        persisted = self._sql_service.execute_in_transaction(
            statements=[delete_stmt],
            rows_to_add=replacement_dtos,
        )
        return [self._from_dto(d) for d in persisted]

    async def _areplace_range(
        self,
        parent_id: ConversationId,
        *,
        up_to_sequence: int,
        replacements: list[StoredMessage],
    ) -> list[StoredMessage]:
        delete_stmt = delete(ChatHistoryMessageDTO).where(
            ChatHistoryMessageDTO.conversation_id == parent_id,
            ChatHistoryMessageDTO.sequence <= up_to_sequence,
        )
        replacement_dtos = [self._to_dto(m) for m in replacements]
        persisted = await self._sql_service.aexecute_in_transaction(
            statements=[delete_stmt],
            rows_to_add=replacement_dtos,
        )
        return [self._from_dto(d) for d in persisted]
