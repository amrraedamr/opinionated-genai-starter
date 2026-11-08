from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsTelemetryConfig(Protocol):
    """Satisfied by `TelemetrySettingsMixin`."""

    otel_enabled: bool
    otel_exporter_endpoint: str
    otel_service_name: str
    otel_sample_ratio: float
    otel_capture_content: bool
