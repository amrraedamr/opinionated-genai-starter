"""OpenAI embeddings behind the neutral seam."""

from base.tools.errors import MissingExtraError

try:
    from openai import AsyncOpenAI, OpenAI
    from openai.types import CreateEmbeddingResponse
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("llm", "openai", feature="OpenAIEmbeddingService") from exc

from base.models.definitions.model import ProviderType
from base.models.view_models.embedding_exception import EmbeddingConfigError
from base.models.view_models.model import EmbeddingConfigurations
from base.services.common.meta import KeyedSingletonABC
from base.services.error_handling.embedding.openai import (
    OpenAIEmbeddingErrorHandler,
)
from base.services.text_embedding import AbstractTextEmbeddingService
from base.tools import openai_clients


class OpenAIEmbeddingService(AbstractTextEmbeddingService, metaclass=KeyedSingletonABC):
    """OpenAI-compatible embeddings.

    Works with any provider following the OpenAI API spec: OpenAI, Azure
    OpenAI (v1 surface), vLLM, Ollama, LiteLLM, Together, Groq, etc.
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
            raise EmbeddingConfigError("model_name is required")

        self._config: EmbeddingConfigurations = config
        self._model: str = config.model_name
        self._batch_size: int = config.batch_size or self.DEFAULT_BATCH_SIZE
        self._dimensions: int | None = config.dimensions
        self._client, self._async_client = self._initialize_clients()
        self._error_handler: OpenAIEmbeddingErrorHandler = OpenAIEmbeddingErrorHandler()

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def get_service_name(self) -> str:
        return "openai"

    def _initialize_clients(self) -> tuple[OpenAI, AsyncOpenAI]:
        return openai_clients.create_clients(
            api_key=self._config.api_key,
            api_host=self._config.api_host,
            api_type=self._config.api_type,
            timeout_total=self.DEFAULT_TIMEOUT_TOTAL,
            timeout_connect=self.DEFAULT_TIMEOUT_CONNECT,
        )

    def _request_kwargs(self, texts: list[str]) -> dict[str, object]:
        kwargs: dict[str, object] = {"input": texts, "model": self._model}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        return kwargs

    @staticmethod
    def _to_vectors(response: CreateEmbeddingResponse) -> list[list[float]]:
        """Sort by index: the API does not promise response order."""
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._error_handler.run(
            self._client.embeddings.create, **self._request_kwargs(texts)
        )
        return self._to_vectors(response)

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        response = await self._error_handler.arun(
            self._async_client.embeddings.create, **self._request_kwargs(texts)
        )
        return self._to_vectors(response)
