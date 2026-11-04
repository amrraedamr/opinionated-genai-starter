"""Composition root for chat_api."""

from base.services.configuration import configure
from chat_api.settings import settings

# Register the service's settings before importing anything that reads them.
configure(settings)

from base.app import create_app  # noqa: E402
from base.capabilities.broker import BrokerDependency  # noqa: E402
from base.capabilities.prompts import Prompts  # noqa: E402
from base.capabilities.sql import SQLDependency  # noqa: E402
from base.capabilities.terminal_capture import TerminalCapture  # noqa: E402
from chat_api import routers  # noqa: E402
from chat_api.lifespans import background_tasks_lifespan  # noqa: E402
from chat_api.messaging import BROKER, messaging_lifespan  # noqa: E402
from chat_api.prompts import templates  # noqa: E402
from chat_api.stores.prompt import PromptStore  # noqa: E402

app = create_app(
    settings=settings,
    routers=routers,
    prefix=settings.api_prefix,
    capabilities=[
        TerminalCapture(settings.service_name),
        Prompts(templates=[templates], store=PromptStore()),
        SQLDependency(),
        BrokerDependency(BROKER),
    ],
    cors_origins=settings.cors_origins_list or None,
    lifespan_hooks=[
        messaging_lifespan,
        background_tasks_lifespan,
    ],
)


def start() -> None:
    from base.tools.webserver import run_webserver

    run_webserver(app_path="chat_api.app:app", settings=settings)
