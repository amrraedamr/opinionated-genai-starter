"""The read side of the settings mixins, grouped the same way.

A mixin declares fields on a service's settings; the matching protocol is what
a `base` subsystem requires of them. Keeping the pair together is what stops
two subsystems inventing different contracts under the same name.

Protocols describing collaborators rather than settings — `Capability`,
`OutboxRow`, `SupportsClose` — stay with the code that consumes them.
"""

from base.services.configuration.protocols.azure_openai import SupportsAzureOpenAIConfig
from base.services.configuration.protocols.blob.azure import SupportsAzureBlobConfig
from base.services.configuration.protocols.broker import (
    SupportsBrokerConfig,
    SupportsRedisBrokerConfig,
)
from base.services.configuration.protocols.cache.redis import SupportsRedisConfig
from base.services.configuration.protocols.credentials import SupportsCredentialsConfig
from base.services.configuration.protocols.embedding.azure_openai import (
    SupportsAzureOpenAIEmbeddingConfig,
)
from base.services.configuration.protocols.embedding.openai import (
    SupportsOpenAIEmbeddingConfig,
)
from base.services.configuration.protocols.guardrails import SupportsGuardrailConfig
from base.services.configuration.protocols.idp import SupportsIdpConfig
from base.services.configuration.protocols.keyvault import SupportsKeyVaultConfig
from base.services.configuration.protocols.llm import SupportsLLMConfig
from base.services.configuration.protocols.llm.anthropic import SupportsAnthropicConfig
from base.services.configuration.protocols.llm.azure_openai import (
    SupportsAzureOpenAILLMConfig,
)
from base.services.configuration.protocols.llm.gemini import SupportsGeminiConfig
from base.services.configuration.protocols.llm.openai import SupportsOpenAILLMConfig
from base.services.configuration.protocols.nosql.cosmos import SupportsCosmosConfig
from base.services.configuration.protocols.nosql.mongo import SupportsMongoConfig
from base.services.configuration.protocols.oauth import SupportsOAuthConfig
from base.services.configuration.protocols.oauth.apple import SupportsAppleOAuthConfig
from base.services.configuration.protocols.oauth.facebook import (
    SupportsFacebookOAuthConfig,
)
from base.services.configuration.protocols.oauth.github import SupportsGitHubOAuthConfig
from base.services.configuration.protocols.oauth.google import SupportsGoogleOAuthConfig
from base.services.configuration.protocols.oauth.microsoft import (
    SupportsMicrosoftOAuthConfig,
)
from base.services.configuration.protocols.prompt import SupportsPromptConfig
from base.services.configuration.protocols.rate_limiting import (
    SupportsRateLimitConfig,
)
from base.services.configuration.protocols.rerank import SupportsRerankConfig
from base.services.configuration.protocols.rerank.cohere import SupportsCohereConfig
from base.services.configuration.protocols.server import SupportsServerConfig
from base.services.configuration.protocols.sql import SupportsSQLConfig
from base.services.configuration.protocols.sql.mysql import SupportsMySQLConfig
from base.services.configuration.protocols.sql.postgres import SupportsPostgresConfig
from base.services.configuration.protocols.telemetry import SupportsTelemetryConfig
from base.services.configuration.protocols.tenancy import SupportsTenancyConfig
from base.services.configuration.protocols.token_issuing import (
    SupportsTokenIssuingConfig,
)
from base.services.configuration.protocols.token_verification import (
    SupportsTokenVerificationConfig,
)
from base.services.configuration.protocols.vector_search.azure_search import (
    SupportsAzureSearchConfig,
)
from base.services.configuration.protocols.vector_search.qdrant import (
    SupportsQdrantConfig,
)

__all__ = [
    "SupportsAnthropicConfig",
    "SupportsAppleOAuthConfig",
    "SupportsAzureBlobConfig",
    "SupportsAzureOpenAIConfig",
    "SupportsAzureOpenAIEmbeddingConfig",
    "SupportsAzureOpenAILLMConfig",
    "SupportsAzureSearchConfig",
    "SupportsBrokerConfig",
    "SupportsCohereConfig",
    "SupportsCosmosConfig",
    "SupportsCredentialsConfig",
    "SupportsFacebookOAuthConfig",
    "SupportsGeminiConfig",
    "SupportsGitHubOAuthConfig",
    "SupportsGoogleOAuthConfig",
    "SupportsGuardrailConfig",
    "SupportsIdpConfig",
    "SupportsKeyVaultConfig",
    "SupportsLLMConfig",
    "SupportsMicrosoftOAuthConfig",
    "SupportsMongoConfig",
    "SupportsMySQLConfig",
    "SupportsOAuthConfig",
    "SupportsOpenAIEmbeddingConfig",
    "SupportsOpenAILLMConfig",
    "SupportsPostgresConfig",
    "SupportsPromptConfig",
    "SupportsQdrantConfig",
    "SupportsRateLimitConfig",
    "SupportsRedisBrokerConfig",
    "SupportsRedisConfig",
    "SupportsRerankConfig",
    "SupportsSQLConfig",
    "SupportsServerConfig",
    "SupportsTelemetryConfig",
    "SupportsTenancyConfig",
    "SupportsTokenIssuingConfig",
    "SupportsTokenVerificationConfig",
]
