"""Composition root for identity."""

from base.services.configuration import configure
from identity.settings import settings

# Register the service's settings before importing anything that reads them.
configure(settings)

from base.app import create_app  # noqa: E402
from base.capabilities.broker import BrokerDependency  # noqa: E402
from base.capabilities.sql import SQLDependency  # noqa: E402
from identity import routers  # noqa: E402
from identity.messaging import BROKER, messaging_lifespan  # noqa: E402

app = create_app(
    settings=settings,
    routers=routers,
    prefix=settings.api_prefix,
    capabilities=[SQLDependency(), BrokerDependency(BROKER)],
    lifespan_hooks=[messaging_lifespan],
)


def start() -> None:
    from base.tools.webserver import run_webserver

    run_webserver(app_path="identity.app:app", settings=settings)
