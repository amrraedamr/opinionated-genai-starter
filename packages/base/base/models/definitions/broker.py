import importlib.util
from enum import StrEnum


class BrokerBackend(StrEnum):
    """Event transport this service publishes and subscribes on.

    `MEMORY` is in-process: correct for local development and for tests, and
    never correct across containers, where each replica would own a private
    queue and events would reach no one.
    """

    MEMORY = "memory"
    REDIS = "redis"

    @property
    def extra(self) -> str | None:
        """`base` extra that installs this transport, if it needs one."""
        match self:
            case BrokerBackend.MEMORY:
                return None
            case BrokerBackend.REDIS:
                return "redis"

    @property
    def distribution(self) -> str | None:
        """Distribution that extra pulls in, for an error message to name."""
        match self:
            case BrokerBackend.MEMORY:
                return None
            case BrokerBackend.REDIS:
                return "redis"

    @property
    def required_imports(self) -> tuple[str, ...]:
        """Import names this transport needs. Empty when it needs nothing."""
        match self:
            case BrokerBackend.MEMORY:
                return ()
            case BrokerBackend.REDIS:
                return ("redis",)

    @property
    def missing_imports(self) -> tuple[str, ...]:
        """Imports this transport needs that the running build does not install."""
        return tuple(
            name
            for name in self.required_imports
            if importlib.util.find_spec(name) is None
        )

    @property
    def is_installed(self) -> bool:
        """Whether this build carries what this transport needs."""
        return not self.missing_imports

    @property
    def is_distributed(self) -> bool:
        """Whether this transport carries events between processes."""
        return self is not BrokerBackend.MEMORY


def installed_brokers() -> tuple[BrokerBackend, ...]:
    """Transports the running build can actually use."""
    return tuple(backend for backend in BrokerBackend if backend.is_installed)
