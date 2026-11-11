"""Composition root for mcp_server."""

from base.services.configuration import configure
from mcp_server.settings import settings

# Register the service's settings before importing anything that reads them.
configure(settings)

from base.app import create_app  # noqa: E402
from base.capabilities.mcp_server import McpServer  # noqa: E402
from mcp_server import routers  # noqa: E402
from mcp_server.tools import toolbox  # noqa: E402

app = create_app(
    settings=settings,
    routers=routers,
    prefix=settings.api_prefix,
    capabilities=[McpServer(toolbox, server_name=settings.service_name)],
)


def start() -> None:
    from base.tools.webserver import run_webserver

    run_webserver(app_path="mcp_server.app:app", settings=settings)
