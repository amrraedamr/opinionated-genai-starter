"""Transport selection, shared by every service that publishes or consumes events."""

from base.models.definitions.backends import BackendSetting
from base.models.definitions.broker import BrokerBackend, installed_brokers
from base.services.broker import AbstractBroker
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsBrokerConfig
from base.tools.errors import UnsupportedBackendError


def build_broker(config: SupportsBrokerConfig | None = None) -> AbstractBroker:
    """Construct the transport this service is configured for.

    Each implementation imports lazily, so a service running the in-process
    transport needs no `redis` extra and vice versa.

    Dispatch is exhaustive rather than defaulting: an unhandled transport
    raises instead of quietly returning the in-process one, which would start
    cleanly, report healthy and deliver nothing between containers.
    """
    settings = config or settings_for(SupportsBrokerConfig)
    installed = ", ".join(member.value for member in installed_brokers())

    try:
        backend = BrokerBackend(settings.broker)
    except ValueError as exc:
        raise ValueError(
            f"{BackendSetting.BROKER}={settings.broker!r} is not a transport "
            f"this library knows. "
            f"This build supports: {installed or 'none'}."
        ) from exc

    if not backend.is_installed:
        raise UnsupportedBackendError(
            backend=backend.value,
            extra=backend.extra,
            distribution=backend.distribution,
            missing=backend.missing_imports,
            available=[member.value for member in installed_brokers()],
            setting=BackendSetting.BROKER,
        )

    match backend:
        case BrokerBackend.REDIS:
            from base.services.broker.redis_streams import RedisStreamsBroker

            return RedisStreamsBroker()
        case BrokerBackend.MEMORY:
            from base.services.broker.memory import InMemoryBroker

            return InMemoryBroker()

    raise ValueError(f"No transport implemented for broker {backend.value!r}")
