
from base.tools.errors import MissingExtraError

try:
    from openai import (
        AsyncOpenAI,
        OpenAI,
    )
    from openai.types import CreateEmbeddingResponse
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="OpenAIEmbeddingService") from exc

from base.models.definitions.model import ProviderType
from base.models.view_models.llm_exception import LLMConfigError
from base.models.view_models.model import EmbeddingConfigurations
from base.services.common.meta import KeyedSingleton
from base.tools import openai_clients


class OpenAITextEmbeddingService(metaclass=KeyedSingleton):
    """
    OpenAI-compatible text embedding service.

    Works with any provider following the OpenAI API spec:
    OpenAI, Azure OpenAI (v1 surface), vLLM, Ollama, LiteLLM, Together, Groq, etc.
    """

    DEFAULT_TIMEOUT_TOTAL: float = 60.0
    DEFAULT_TIMEOUT_CONNECT: float = 5.0
    DEFAULT_BATCH_SIZE: int = 2048

    def __init__(self, config: EmbeddingConfigurations | None = None) -> None:
        if not config:
            config = EmbeddingConfigurations.get_default_for_provider(
                ProviderType.AZURE_OPENAI
            )

        if config.model_name is None:
            raise LLMConfigError("model_name is required")

        self._config: EmbeddingConfigurations = config
        self._model: str = config.model_name
        self._batch_size: int = config.batch_size or self.DEFAULT_BATCH_SIZE
        self._dimensions: int | None = config.dimensions

        self._client, self._async_client = self._initialize_clients()

    def _initialize_clients(self) -> tuple[OpenAI, AsyncOpenAI]:
        return openai_clients.create_clients(
            api_key=self._config.api_key,
            api_host=self._config.api_host,
            api_type=self._config.api_type,
            timeout_total=self.DEFAULT_TIMEOUT_TOTAL,
            timeout_connect=self.DEFAULT_TIMEOUT_CONNECT,
        )

    # Internals

    def _create(self, input: str | list[str]) -> CreateEmbeddingResponse:
        if self._dimensions is not None:
            return self._client.embeddings.create(
                input=input, model=self._model, dimensions=self._dimensions
            )
        return self._client.embeddings.create(input=input, model=self._model)

    async def _acreate(self, input: str | list[str]) -> CreateEmbeddingResponse:
        if self._dimensions is not None:
            return await self._async_client.embeddings.create(
                input=input, model=self._model, dimensions=self._dimensions
            )
        return await self._async_client.embeddings.create(
            input=input, model=self._model
        )

    # Sync

    def embed_query(self, text: str) -> list[float]:
        return self._create(text).data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch: list[str] = texts[i : i + self._batch_size]
            response = self._create(batch)
            results.extend(
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            )
        return results

    # Async

    async def aembed_query(self, text: str) -> list[float]:
        return (await self._acreate(text)).data[0].embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch: list[str] = texts[i : i + self._batch_size]
            response = await self._acreate(batch)
            results.extend(
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            )
        return results
