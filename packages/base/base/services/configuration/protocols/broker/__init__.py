from typing import Protocol, runtime_checkable

from base.models.definitions.broker import BrokerBackend
from base.services.configuration.protocols.cache.redis import SupportsRedisConfig


@runtime_checkable
class SupportsBrokerConfig(Protocol):
    """Satisfied by `BrokerSettingsMixin`."""

    broker: BrokerBackend
    broker_max_attempts: int
    broker_claim_idle_ms: int
    broker_retry_backoff_seconds: float


@runtime_checkable
class SupportsRedisBrokerConfig(SupportsRedisConfig, SupportsBrokerConfig, Protocol):
    """What a Redis-backed broker reads: the connection plus the delivery policy."""
