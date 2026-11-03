from base.services.logging.console import ConsoleLoggerService
from contracts.events import Event

logger = ConsoleLoggerService()
_processed_count = 0


async def ahandle_user_registered(event: Event) -> None:
    global _processed_count
    _processed_count += 1
    logger.info(
        "user registered event processed: user_id=%s correlation=%s",
        event.payload.get("user_id"),
        event.meta.correlation_id,
    )


async def ahandle_conversation_created(event: Event) -> None:
    global _processed_count
    _processed_count += 1
    logger.info(
        "conversation created event processed: conversation_id=%s correlation=%s",
        event.payload.get("conversation_id"),
        event.meta.correlation_id,
    )


def processed_count() -> int:
    return _processed_count
