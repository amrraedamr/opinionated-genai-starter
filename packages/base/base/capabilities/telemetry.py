"""Tracing, configured per worker and flushed on the way out."""

from base.tools.errors import MissingExtraError

try:
    from fastapi import FastAPI
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("server", "fastapi", feature="Telemetry") from exc

from base.models.definitions.health import ProbeStatus
from base.models.view_models.health import ProbeResult
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsTelemetryConfig
from base.services.logging.console import ConsoleLoggerService
from base.services.telemetry import Telemetry as TelemetryService


class Telemetry:
    """Exports traces over OTLP while this service is running.

    The provider is built in `astartup` rather than at import: gunicorn
    preloads the application and then forks, and an exporter built before the
    fork would be shared by every worker.
    """

    name = "telemetry"

    def __init__(self, config: SupportsTelemetryConfig | None = None) -> None:
        self._config = config
        self._logger = ConsoleLoggerService()

    async def astartup(self, app: FastAPI) -> None:
        config = self._config or settings_for(SupportsTelemetryConfig)
        TelemetryService().configure(config, app=app)
        if config.otel_enabled:
            self._logger.info(
                "Tracing enabled, exporting to %s", config.otel_exporter_endpoint
            )

    async def ashutdown(self, app: FastAPI) -> None:
        del app
        TelemetryService().shutdown()

    async def aprobe(self) -> ProbeResult:
        """Readiness never depends on the collector: traces are not the service."""
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.OK,
            detail="enabled" if TelemetryService().enabled else "disabled",
        )
