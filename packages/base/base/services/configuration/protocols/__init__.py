"""The read side of the settings mixins, one file per mixin.

A mixin declares fields on a service's settings; the matching protocol is what
a `base` subsystem requires of them. Keeping the pair together is what stops
two subsystems inventing different contracts under the same name.

Protocols describing collaborators rather than settings — `Capability`,
`OutboxRow`, `SupportsClose` — stay with the code that consumes them.
"""

from base.services.configuration.protocols.azure_search import (
    SupportsAzureSearchConfig,
)
from base.services.configuration.protocols.broker import SupportsBrokerConfig
from base.services.configuration.protocols.credentials import (
    SupportsCredentialsConfig,
)
from base.services.configuration.protocols.embedding import SupportsEmbeddingConfig
from base.services.configuration.protocols.idp import SupportsIdpConfig
from base.services.configuration.protocols.llm import SupportsLLMConfig
from base.services.configuration.protocols.mysql import SupportsMySQLConfig
from base.services.configuration.protocols.postgres import SupportsPostgresConfig
from base.services.configuration.protocols.prompt import SupportsPromptConfig
from base.services.configuration.protocols.qdrant import SupportsQdrantConfig
from base.services.configuration.protocols.redis import SupportsRedisConfig
from base.services.configuration.protocols.rerank import (
    SupportsCohereConfig,
    SupportsRerankConfig,
)
from base.services.configuration.protocols.server import SupportsServerConfig
from base.services.configuration.protocols.sql import SupportsSQLConfig
from base.services.configuration.protocols.token_issuing import (
    SupportsTokenIssuingConfig,
)
from base.services.configuration.protocols.token_verification import (
    SupportsTokenVerificationConfig,
)

__all__ = [
    "SupportsCohereConfig",
    "SupportsRerankConfig",
    "SupportsAzureSearchConfig",
    "SupportsBrokerConfig",
    "SupportsCredentialsConfig",
    "SupportsEmbeddingConfig",
    "SupportsIdpConfig",
    "SupportsLLMConfig",
    "SupportsMySQLConfig",
    "SupportsPostgresConfig",
    "SupportsPromptConfig",
    "SupportsQdrantConfig",
    "SupportsRedisConfig",
    "SupportsSQLConfig",
    "SupportsServerConfig",
    "SupportsTokenIssuingConfig",
    "SupportsTokenVerificationConfig",
]
