"""Settings a service composes, grouped by the capability they configure.

A capability holds what no provider owns in its package, with a module per
provider beside it, so adding a provider is a file rather than a rename and a
service declares the providers it configures and nothing else.
"""

from base.services.configuration.mixins.azure_openai import AzureOpenAISettingsMixin
from base.services.configuration.mixins.blob.azure import AzureBlobSettingsMixin
from base.services.configuration.mixins.broker import BrokerSettingsMixin
from base.services.configuration.mixins.cache.redis import RedisSettingsMixin
from base.services.configuration.mixins.credentials import CredentialsSettingsMixin
from base.services.configuration.mixins.embedding.azure_openai import (
    AzureOpenAIEmbeddingSettingsMixin,
)
from base.services.configuration.mixins.embedding.openai import (
    OpenAIEmbeddingSettingsMixin,
)
from base.services.configuration.mixins.idp import IdpSettingsMixin
from base.services.configuration.mixins.keyvault import KeyVaultSettingsMixin
from base.services.configuration.mixins.llm import LLMSettingsMixin
from base.services.configuration.mixins.llm.anthropic import AnthropicSettingsMixin
from base.services.configuration.mixins.llm.azure_openai import (
    AzureOpenAILLMSettingsMixin,
)
from base.services.configuration.mixins.llm.gemini import GeminiSettingsMixin
from base.services.configuration.mixins.llm.openai import OpenAILLMSettingsMixin
from base.services.configuration.mixins.nosql.cosmos import CosmosSettingsMixin
from base.services.configuration.mixins.nosql.mongo import MongoSettingsMixin
from base.services.configuration.mixins.oauth import OAuthSettingsMixin
from base.services.configuration.mixins.oauth.apple import AppleOAuthSettingsMixin
from base.services.configuration.mixins.oauth.facebook import FacebookOAuthSettingsMixin
from base.services.configuration.mixins.oauth.github import GitHubOAuthSettingsMixin
from base.services.configuration.mixins.oauth.google import GoogleOAuthSettingsMixin
from base.services.configuration.mixins.oauth.microsoft import (
    MicrosoftOAuthSettingsMixin,
)
from base.services.configuration.mixins.prompt import PromptSettingsMixin
from base.services.configuration.mixins.rerank import RerankSettingsMixin
from base.services.configuration.mixins.rerank.cohere import CohereSettingsMixin
from base.services.configuration.mixins.server import ServerSettingsMixin
from base.services.configuration.mixins.sql import SQLSettingsMixin
from base.services.configuration.mixins.sql.mysql import MySQLSettingsMixin
from base.services.configuration.mixins.sql.postgres import PostgresSettingsMixin
from base.services.configuration.mixins.telemetry import TelemetrySettingsMixin
from base.services.configuration.mixins.tenancy import TenancySettingsMixin
from base.services.configuration.mixins.token_issuing import TokenIssuingSettingsMixin
from base.services.configuration.mixins.token_verification import (
    TokenVerificationSettingsMixin,
)
from base.services.configuration.mixins.vector_search.azure_search import (
    AzureSearchSettingsMixin,
)
from base.services.configuration.mixins.vector_search.qdrant import QdrantSettingsMixin

__all__ = [
    "AnthropicSettingsMixin",
    "AppleOAuthSettingsMixin",
    "AzureBlobSettingsMixin",
    "AzureOpenAIEmbeddingSettingsMixin",
    "AzureOpenAILLMSettingsMixin",
    "AzureOpenAISettingsMixin",
    "AzureSearchSettingsMixin",
    "BrokerSettingsMixin",
    "CohereSettingsMixin",
    "CosmosSettingsMixin",
    "CredentialsSettingsMixin",
    "FacebookOAuthSettingsMixin",
    "GeminiSettingsMixin",
    "GitHubOAuthSettingsMixin",
    "GoogleOAuthSettingsMixin",
    "IdpSettingsMixin",
    "KeyVaultSettingsMixin",
    "LLMSettingsMixin",
    "MicrosoftOAuthSettingsMixin",
    "MongoSettingsMixin",
    "MySQLSettingsMixin",
    "OAuthSettingsMixin",
    "OpenAIEmbeddingSettingsMixin",
    "OpenAILLMSettingsMixin",
    "PostgresSettingsMixin",
    "PromptSettingsMixin",
    "QdrantSettingsMixin",
    "RedisSettingsMixin",
    "RerankSettingsMixin",
    "SQLSettingsMixin",
    "ServerSettingsMixin",
    "TelemetrySettingsMixin",
    "TenancySettingsMixin",
    "TokenIssuingSettingsMixin",
    "TokenVerificationSettingsMixin",
]
