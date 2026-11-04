"""Cohere Rerank, behind the `rerank` extra."""

from base.tools.errors import MissingExtraError

try:
    from cohere import AsyncClientV2, ClientV2
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("rerank", "cohere", feature="CohereRerankService") from exc

from base.models.view_models.rerank import RerankResult
from base.services.common.meta import SingletonABC
from base.services.configuration import get_settings
from base.services.configuration.protocols.rerank import SupportsCohereConfig
from base.services.reranking import AbstractRerankService


class CohereRerankService(AbstractRerankService, metaclass=SingletonABC):
    """Reranking through Cohere's hosted cross-encoder."""

    def __init__(self, config: SupportsCohereConfig | None = None) -> None:
        super().__init__()
        settings = config or get_settings()
        self._model: str = settings.cohere_rerank_model
        self._api_key: str = settings.cohere_api_key
        self._sync_client: ClientV2 | None = None
        self._async_client: AsyncClientV2 | None = None

    @property
    def _client(self) -> ClientV2:
        if self._sync_client is None:
            self._sync_client = ClientV2(api_key=self._api_key)
        return self._sync_client

    @property
    def _aclient(self) -> AsyncClientV2:
        if self._async_client is None:
            self._async_client = AsyncClientV2(api_key=self._api_key)
        return self._async_client

    @staticmethod
    def _to_results(response: object, documents: list[str]) -> list[RerankResult]:
        return [
            RerankResult(
                index=item.index,
                score=float(item.relevance_score),
                document=documents[item.index],
            )
            for item in getattr(response, "results", [])
        ]

    def _rerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> list[RerankResult]:
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n or len(documents),
        )
        return self._to_results(response, documents)

    async def _arerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> list[RerankResult]:
        response = await self._aclient.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n or len(documents),
        )
        return self._to_results(response, documents)

    def get_service_name(self) -> str:
        return "cohere-rerank"
