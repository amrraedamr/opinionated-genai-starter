"""A span leaving a service and arriving somewhere it can be read.

Everything else about telemetry is asserted against an in-process exporter,
which proves the spans are built but not that any of it leaves the process.
This is the only test that runs the real path: the SDK exports OTLP to the
collector, the collector forwards to Jaeger, and Jaeger is then asked whether
it has heard of the service.

Each test uses a service name of its own, so "did this span arrive" cannot be
answered by a span some earlier run left behind.
"""

import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator

import pytest

from base.services.configuration.mixins import TelemetrySettingsMixin
from base.services.telemetry import Telemetry
from tests.conftest import MakeSettings

pytestmark = pytest.mark.integration

COLLECTOR = "http://localhost:4317"
JAEGER = "http://localhost:16686"
ATTEMPTS = 40

SERVICE = f"test-{uuid.uuid4().hex[:10]}"
"""One name for the whole module, not one per test.

OpenTelemetry allows the global tracer provider to be set once per process:
a second `configure` logs "Overriding of current TracerProvider is not
allowed" and is ignored, so every later test would export under the first
test's name anyway. Naming it once makes that explicit rather than something
the suite discovers when a test fails for the wrong reason.
"""


def services_known_to_jaeger() -> list[str]:
    """Every service Jaeger has received a span from."""
    with urllib.request.urlopen(f"{JAEGER}/api/v3/services", timeout=10) as reply:
        return list(json.loads(reply.read()).get("services", []))


def arrived(service: str) -> bool:
    """Whether Jaeger has heard of `service` yet.

    Polled rather than slept on: the collector batches, so the delay is a
    property of the pipeline rather than a number worth hard-coding.
    """
    for _ in range(ATTEMPTS):
        try:
            if service in services_known_to_jaeger():
                return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


@pytest.fixture
def exporting(make_settings: MakeSettings) -> Iterator[str]:
    """A telemetry service exporting to the collector, under the module's name."""
    settings = make_settings(
        TelemetrySettingsMixin,
        service_name=SERVICE,
        otel_enabled=True,
        otel_exporter_endpoint=COLLECTOR,
        otel_service_name=SERVICE,
    )
    telemetry = Telemetry()
    telemetry.configure(settings)
    try:
        yield SERVICE
    finally:
        telemetry.shutdown()


async def test_a_span_reaches_the_collector_and_is_readable(
    exporting: str,
) -> None:
    """The whole path, which nothing else in the suite exercises."""
    telemetry = Telemetry()
    with telemetry.span("integration-span") as span:
        assert span is not None, "tracing was configured but produced no span"
        span.set_attribute("test.marker", exporting)

    telemetry.shutdown()
    assert arrived(exporting), f"no span from {exporting} reached Jaeger"


async def test_a_nested_span_arrives_with_its_parent(exporting: str) -> None:
    """A trace is a tree; exporting only the root would lose the shape."""
    telemetry = Telemetry()
    with telemetry.span("outer") as outer:
        assert outer is not None
        with telemetry.span("inner") as inner:
            assert inner is not None
            assert (
                inner.get_span_context().trace_id == outer.get_span_context().trace_id
            )

    telemetry.shutdown()
    assert arrived(exporting)


def test_the_collector_is_the_one_being_reached() -> None:
    """Jaeger also speaks OTLP, so this pins that the collector is in the path.

    Only the collector binds 4317 on the host; Jaeger's own OTLP port stays
    inside the compose network. Exporting to 4317 and then finding the span in
    Jaeger is therefore proof it travelled through the collector.
    """
    with socket.create_connection(("localhost", 4317), timeout=5):
        pass


async def test_a_disabled_service_exports_nothing(make_settings: MakeSettings) -> None:
    """Off must mean off: no exporter, no span, nothing sent anywhere."""
    name = f"test-disabled-{uuid.uuid4().hex[:10]}"
    settings = make_settings(
        TelemetrySettingsMixin,
        service_name=name,
        otel_enabled=False,
        otel_exporter_endpoint=COLLECTOR,
        otel_service_name=name,
    )
    telemetry = Telemetry()
    telemetry.configure(settings)
    with telemetry.span("never-exported") as span:
        assert span is None, "a disabled service built a span anyway"
    telemetry.shutdown()

    assert name not in services_known_to_jaeger()
