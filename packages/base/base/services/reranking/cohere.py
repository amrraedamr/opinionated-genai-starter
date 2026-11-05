"""Cohere Rerank, behind the `rerank` extra."""

from base.tools.errors import MissingExtraError

try:
    from cohere import AsyncClientV2, ClientV2
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("rerank", "cohere", feature="CohereRerankService") from exc

from base.models.view_models.exceptions.rerank import RerankConfigError
from base.models.view_models.rerank import RerankBatch, RerankResult
from base.models.view_models.usage import ProviderUsage
from base.services.common.meta import SingletonABC
from base.services.configuration import get_settings
from base.services.configuration.protocols import (
    SupportsCohereConfig,
)
from base.services.error_handling import aretrying, retrying
from base.services.error_handling.rerank.cohere import CohereErrorHandler
from base.services.reranking import AbstractRerankService


class CohereRerankService(AbstractRerankService, metaclass=SingletonABC):
    """Reranking through Cohere's hosted cross-encoder."""

    def __init__(self, config: SupportsCohereConfig | None = None) -> None:
        super().__init__()
        settings = config or get_settings()
        self._model: str = settings.cohere_rerank_model
        self._api_key: str = settings.cohere_api_key
        if not self._model:
            raise RerankConfigError("cohere_rerank_model is required")
        self._sync_client: ClientV2 | None = None
        self._async_client: AsyncClientV2 | None = None
        self._error_handler: CohereErrorHandler = CohereErrorHandler()

    @property
    def model(self) -> str:
        return self._model

    @property
    def _client(self) -> ClientV2:
        # SDK-level retries are off: CohereErrorHandler owns retry policy, and
        # two layers of backoff would multiply the wait.
        if self._sync_client is None:
            self._sync_client = ClientV2(api_key=self._api_key, max_retries=0)
        return self._sync_client

    @property
    def _aclient(self) -> AsyncClientV2:
        if self._async_client is None:
            self._async_client = AsyncClientV2(api_key=self._api_key, max_retries=0)
        return self._async_client

    @staticmethod
    def _to_batch(response: object, documents: list[str]) -> RerankBatch:
        """Cohere bills reranking in search units, and reports tokens beside them."""
        meta = getattr(response, "meta", None)
        tokens = getattr(meta, "tokens", None) if meta else None
        return RerankBatch(
            results=[
                RerankResult(
                    index=item.index,
                    score=float(item.relevance_score),
                    document=documents[item.index],
                )
                for item in getattr(response, "results", [])
            ],
            usage=ProviderUsage(
                input_tokens=int(getattr(tokens, "input_tokens", 0) or 0),
                output_tokens=int(getattr(tokens, "output_tokens", 0) or 0),
                cached_input_tokens=int(getattr(meta, "cached_tokens", 0) or 0),
            )
            if meta
            else None,
        )

    @retrying
    def _rerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> RerankBatch:
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n or len(documents),
        )
        return self._to_batch(response, documents)

    @aretrying
    async def _arerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> RerankBatch:
        response = await self._aclient.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n or len(documents),
        )
        return self._to_batch(response, documents)

    def get_service_name(self) -> str:
        return "cohere-rerank"
