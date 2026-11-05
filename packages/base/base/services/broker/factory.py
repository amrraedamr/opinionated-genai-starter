"""Transport selection, shared by every service that publishes or consumes events."""

from base.models.definitions.broker import BrokerBackend
from base.services.broker import AbstractBroker
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsBrokerConfig


def build_broker(config: SupportsBrokerConfig | None = None) -> AbstractBroker:
    """Construct the transport this service is configured for.

    Each implementation imports lazily, so a service running the in-process
    transport needs no `redis` extra and vice versa.

    Dispatch is exhaustive rather than defaulting: an unhandled transport
    raises instead of quietly returning the in-process one, which would start
    cleanly, report healthy and deliver nothing between containers.
    """
    settings = config or settings_for(SupportsBrokerConfig)
    backend = BrokerBackend(settings.broker)

    match backend:
        case BrokerBackend.REDIS:
            from base.services.broker.redis_streams import RedisStreamsBroker

            return RedisStreamsBroker()
        case BrokerBackend.MEMORY:
            from base.services.broker.memory import InMemoryBroker

            return InMemoryBroker()

    raise ValueError(f"No transport implemented for broker {backend.value!r}")
