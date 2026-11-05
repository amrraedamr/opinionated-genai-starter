"""One settings mixin per optional subsystem, mirroring the dependency extras.

A service composes `CoreSettings` with only the mixins it actually uses, so a
worker with no HTTP surface never carries HTTP settings.
"""

from base.services.configuration.mixins.anthropic import AnthropicSettingsMixin
from base.services.configuration.mixins.azure_blob import AzureBlobSettingsMixin
from base.services.configuration.mixins.azure_openai import (
    AzureOpenAIEmbeddingSettingsMixin,
    AzureOpenAILLMSettingsMixin,
    AzureOpenAISettingsMixin,
)
from base.services.configuration.mixins.azure_search import AzureSearchSettingsMixin
from base.services.configuration.mixins.broker import BrokerSettingsMixin
from base.services.configuration.mixins.cohere import CohereSettingsMixin
from base.services.configuration.mixins.cosmos import CosmosSettingsMixin
from base.services.configuration.mixins.credentials import CredentialsSettingsMixin
from base.services.configuration.mixins.idp import IdpResourceServerMixin
from base.services.configuration.mixins.keyvault import KeyVaultSettingsMixin
from base.services.configuration.mixins.llm import LLMSettingsMixin
from base.services.configuration.mixins.mongo import MongoSettingsMixin
from base.services.configuration.mixins.mysql import MySQLSettingsMixin
from base.services.configuration.mixins.oauth import OAuthSettingsMixin
from base.services.configuration.mixins.openai import (
    OpenAIEmbeddingSettingsMixin,
    OpenAILLMSettingsMixin,
)
from base.services.configuration.mixins.postgres import PostgresSettingsMixin
from base.services.configuration.mixins.prompt import PromptSettingsMixin
from base.services.configuration.mixins.qdrant import QdrantSettingsMixin
from base.services.configuration.mixins.redis import RedisSettingsMixin
from base.services.configuration.mixins.rerank import RerankSettingsMixin
from base.services.configuration.mixins.server import ServerSettingsMixin
from base.services.configuration.mixins.sql import SQLSettingsMixin
from base.services.configuration.mixins.token_issuing import TokenIssuingMixin
from base.services.configuration.mixins.token_verification import TokenVerificationMixin

__all__ = [
    "CohereSettingsMixin",
    "RerankSettingsMixin",
    "AzureSearchSettingsMixin",
    "AzureBlobSettingsMixin",
    "BrokerSettingsMixin",
    "CosmosSettingsMixin",
    "CredentialsSettingsMixin",
    "IdpResourceServerMixin",
    "KeyVaultSettingsMixin",
    "AnthropicSettingsMixin",
    "AzureOpenAIEmbeddingSettingsMixin",
    "AzureOpenAILLMSettingsMixin",
    "AzureOpenAISettingsMixin",
    "LLMSettingsMixin",
    "OpenAIEmbeddingSettingsMixin",
    "OpenAILLMSettingsMixin",
    "MongoSettingsMixin",
    "MySQLSettingsMixin",
    "OAuthSettingsMixin",
    "PostgresSettingsMixin",
    "PromptSettingsMixin",
    "QdrantSettingsMixin",
    "RedisSettingsMixin",
    "SQLSettingsMixin",
    "ServerSettingsMixin",
    "TokenIssuingMixin",
    "TokenVerificationMixin",
]
