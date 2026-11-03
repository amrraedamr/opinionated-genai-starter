# Import each table so it registers on ChatBase.metadata. Add a line per table.
import chat_api.models.orm.chat_history_message  # noqa: F401
import chat_api.models.orm.conversation  # noqa: F401
import chat_api.models.orm.outbox  # noqa: F401
import chat_api.models.orm.prompt  # noqa: F401
from base.tools.migrations import run_migrations
from chat_api.models.orm import ChatBase
from chat_api.settings import settings

run_migrations(ChatBase.metadata, settings)
