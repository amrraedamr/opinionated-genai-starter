from pydantic import BaseModel


class TelemetrySettingsMixin(BaseModel):
    """Export traces over OTLP. Off unless a service turns it on."""

    otel_enabled: bool = False
    otel_exporter_endpoint: str = "http://localhost:4317"
    otel_service_name: str = ""
    otel_sample_ratio: float = 1.0
    otel_capture_content: bool = False
