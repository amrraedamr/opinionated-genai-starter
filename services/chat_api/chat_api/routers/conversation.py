from typing import Any
from uuid import UUID

from fastapi import APIRouter
from sse_starlette import EventSourceResponse

from base.models.definitions.errors import DatabaseErrors
from base.models.definitions.message import MessageRole
from base.models.view_models import BaseResponse
from base.models.view_models.chat_history import ChatHistoryConfig, ConversationId
from base.models.view_models.execution_context import FlowVariables
from base.models.view_models.flow import FlowAbortError
from base.models.view_models.streams import StreamError
from base.services.auth import Principal, principal_dependency
from base.services.background_tasks.in_memory import InMemoryBackgroundTaskService
from base.services.chat_history import ChatHistoryService
from base.services.common.meta import Stored, scope_id
from base.services.execution_context import ExecutionContext
from base.services.logging.console import ConsoleLoggerService
from base.tools.errors import ErrorsException, raise_error
from base.tools.locale import get_text_locale
from base.tools.stream import StreamIterator
from chat_api.models.view_models.chat import (
    ChatMessageView,
    ListChatResponse,
    SendChatRequest,
    SendChatResponse,
)
from chat_api.models.view_models.conversation import (
    CreateConversationRequest,
    CreateConversationResponse,
    DeleteConversationResponse,
    EditConversationRequest,
    EditConversationResponse,
    GetConversationResponse,
)
from chat_api.services.agents.test import TestAgent
from chat_api.stores.chat_history.sql import SqlChatHistoryStore
from chat_api.stores.conversation import ConversationStore

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post("", response_model=BaseResponse[CreateConversationResponse])
async def create_conversation(
    request: CreateConversationRequest,
    principal: Principal = principal_dependency,
) -> BaseResponse[CreateConversationResponse]:
    created = await ConversationStore().acreate_for_user(principal.user_id, request)
    return BaseResponse(data=CreateConversationResponse(conversation_id=created.id))


@router.get("/{conversation_id}", response_model=BaseResponse[GetConversationResponse])
async def get_conversation(
    conversation_id: UUID,
    principal: Principal = principal_dependency,
) -> BaseResponse[GetConversationResponse]:
    conversation = await ConversationStore().aget_for_user(
        principal.user_id, conversation_id
    )
    if conversation is None:
        raise_error(DatabaseErrors.RECORD_NOT_FOUND)
    assert conversation is not None

    return BaseResponse(
        data=GetConversationResponse(
            conversation_id=conversation.id,
            title=conversation.title,
            metadata=conversation.meta,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
    )


@router.patch(
    "/{conversation_id}", response_model=BaseResponse[EditConversationResponse]
)
async def edit_conversation(
    conversation_id: UUID,
    request: EditConversationRequest,
    principal: Principal = principal_dependency,
) -> BaseResponse[EditConversationResponse]:
    updated = await ConversationStore().aedit_for_user(
        principal.user_id,
        conversation_id,
        request,
    )
    if updated is None:
        raise_error(DatabaseErrors.RECORD_NOT_FOUND)

    return BaseResponse(data=EditConversationResponse())


@router.delete(
    "/{conversation_id}", response_model=BaseResponse[DeleteConversationResponse]
)
async def delete_conversation(
    conversation_id: UUID,
    principal: Principal = principal_dependency,
) -> BaseResponse[DeleteConversationResponse]:
    deleted = await ConversationStore().adelete_for_user(
        principal.user_id, conversation_id
    )
    if not deleted:
        raise_error(DatabaseErrors.RECORD_NOT_FOUND)

    return BaseResponse(data=DeleteConversationResponse())


@router.post(
    "/{conversation_id}/messages", response_model=BaseResponse[SendChatResponse]
)
async def send_message(
    conversation_id: UUID,
    request: SendChatRequest,
    streaming: bool = False,
    debug: bool = False,
    principal: Principal = principal_dependency,
) -> BaseResponse[SendChatResponse] | EventSourceResponse:
    execution_context = ExecutionContext()
    if debug:
        # Starting the trace is what makes the adapters capture prompts, so an
        # ordinary request pays nothing for a feature it did not ask for.
        execution_context.start_debug_trace()

    chat_history_service = ChatHistoryService(ChatHistoryConfig(), SqlChatHistoryStore)
    logger = ConsoleLoggerService()

    conversation = await ConversationStore().aget_for_user(
        principal.user_id, conversation_id
    )
    if conversation is None:
        raise_error(DatabaseErrors.RECORD_NOT_FOUND)
    assert conversation is not None
    conversation_key = ConversationId(conversation.id)

    execution_context.flow_variables = FlowVariables(
        conversation_id=conversation_key,
        user_query=request.content,
        locale=get_text_locale(request.content),
    )

    stream_iterator: StreamIterator | None = None
    request_scope_id = scope_id.get()

    async def run() -> str:
        try:
            await chat_history_service.aappend_message(
                conversation_key,
                role=MessageRole.USER,
                content=request.content,
            )

            await TestAgent().arun()

            metadata: dict[str, Any] = {}

            await chat_history_service.aappend_message(
                conversation_key,
                role=MessageRole.ASSISTANT,
                content=execution_context.flow_variables.final_answer,
                tool_calls=[],
                metadata=metadata,
            )

            return execution_context.flow_variables.final_answer
        except FlowAbortError as e:
            return e.canned_message.value[execution_context.flow_variables.locale]
        except ErrorsException as e:
            logger.error(f"send_message run failed: {e}")
            if stream_iterator is not None:
                await stream_iterator.asend(
                    StreamError(
                        error_code=e.status_code or 500,
                        text=str(e),
                    )
                )
            raise
        except Exception as e:
            # SSE boundary only: emit an error frame so clients don't hang.
            if stream_iterator is None:
                raise
            logger.error(f"send_message run failed: {e}")
            await stream_iterator.asend(
                StreamError(error_code=500, text="Internal server error")
            )
            raise
        finally:
            Stored.delete_scope(request_scope_id)

    if streaming:
        stream_iterator = StreamIterator()
        execution_context.streaming = True
        execution_context.stream_iterator = stream_iterator
        InMemoryBackgroundTaskService().spawn_coroutine(
            run(), name="send_message.stream"
        )
        return EventSourceResponse(stream_iterator, media_type="text/event-stream")

    result = await run()
    # The scope is gone by now, but this reference still holds the trace.
    return BaseResponse(
        data=SendChatResponse(
            content=result or "",
            debug=execution_context.debug_trace if debug else None,
        )
    )


@router.get(
    "/{conversation_id}/messages", response_model=BaseResponse[ListChatResponse]
)
async def get_messages(
    conversation_id: UUID,
    principal: Principal = principal_dependency,
) -> BaseResponse[ListChatResponse]:
    conversation = await ConversationStore().aget_for_user(
        principal.user_id, conversation_id
    )
    if conversation is None:
        raise_error(DatabaseErrors.RECORD_NOT_FOUND)
    assert conversation is not None
    conversation_key = ConversationId(conversation.id)
    stored = await SqlChatHistoryStore().aget(conversation_key)

    return BaseResponse(
        data=ListChatResponse(
            messages=[
                ChatMessageView(**message.serialize(by_alias=False))
                for message in stored
            ]
        )
    )
