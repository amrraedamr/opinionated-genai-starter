import json
from typing import Any

from base.tools.errors import MissingExtraError

try:
    from azure.core.exceptions import AzureError, ResourceNotFoundError
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchAlgorithmMetric,
        VectorSearchProfile,
    )
    from azure.search.documents.models import VectorizedQuery
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "azure", "azure-search-documents", feature="AzureAISearchService"
    ) from exc

from base.clients.vectors.azure_ai_search import AzureAISearchClient
from base.models.definitions.vector_search import (
    VECTOR_UPSERT_BATCH_SIZE,
    DistanceMetric,
)
from base.models.view_models.vector_search import VectorRecord, VectorSearchResult
from base.services.common.meta import SingletonABC
from base.services.configuration import get_settings
from base.services.vector_search import AbstractVectorSearchService
from base.tools.errors import ServiceHealthCheckError

_METRIC_MAP: dict[str, VectorSearchAlgorithmMetric] = {
    DistanceMetric.COSINE: VectorSearchAlgorithmMetric.COSINE,
    DistanceMetric.DOT: VectorSearchAlgorithmMetric.DOT_PRODUCT,
    DistanceMetric.EUCLIDEAN: VectorSearchAlgorithmMetric.EUCLIDEAN,
}


class AzureAISearchService(AbstractVectorSearchService, metaclass=SingletonABC):
    """Azure AI Search-backed vector search service with sync and async support.

    Document schema convention:
        - ``id`` -- key field (string, filterable).
        - ``vector`` -- vector field (Collection(Edm.Single)).
        - ``metadata`` -- JSON-serialized metadata string.

    Extra top-level filterable fields can be declared when creating the index
    via the ``extra_fields`` kwarg (a list of ``SearchField`` / ``SimpleField``
    objects).
    """

    def __init__(self) -> None:
        super().__init__()
        self._client: AzureAISearchClient = AzureAISearchClient()
        self._vector_field: str = get_settings().azure_search_vector_field
        self._key_field: str = get_settings().azure_search_key_field

    # Collection / Index Management

    def create_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        **kwargs: Any,
    ) -> None:
        index = self._build_index(collection, vector_size, distance, **kwargs)
        self._client.sync_index_client.create_or_update_index(index)

    async def acreate_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        **kwargs: Any,
    ) -> None:
        index = self._build_index(collection, vector_size, distance, **kwargs)
        await self._client.async_index_client.create_or_update_index(index)

    def delete_collection(self, collection: str) -> None:
        self._client.sync_index_client.delete_index(collection)

    async def adelete_collection(self, collection: str) -> None:
        await self._client.async_index_client.delete_index(collection)

    def collection_exists(self, collection: str) -> bool:
        names: list[str] = list(self._client.sync_index_client.list_index_names())
        return collection in names

    async def acollection_exists(self, collection: str) -> bool:
        names: list[str] = [
            name async for name in self._client.async_index_client.list_index_names()
        ]
        return collection in names

    # Upsert

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        client = self._client.get_search_client(collection)
        for i in range(0, len(records), VECTOR_UPSERT_BATCH_SIZE):
            batch = records[i : i + VECTOR_UPSERT_BATCH_SIZE]
            documents = [self._record_to_document(r) for r in batch]
            client.merge_or_upload_documents(documents)

    async def aupsert(self, collection: str, records: list[VectorRecord]) -> None:
        client = self._client.get_async_search_client(collection)
        for i in range(0, len(records), VECTOR_UPSERT_BATCH_SIZE):
            batch = records[i : i + VECTOR_UPSERT_BATCH_SIZE]
            documents = [self._record_to_document(r) for r in batch]
            await client.merge_or_upload_documents(documents)

    # Search

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        client = self._client.get_search_client(collection)
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=limit,
            fields=self._vector_field,
        )
        results = client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=self._build_filter(filters),
            top=limit,
        )
        output: list[VectorSearchResult] = []
        for doc in results:
            score: float = float(doc.get("@search.score", 0.0))
            if score_threshold is not None and score < score_threshold:
                continue
            output.append(self._document_to_search_result(doc, score))
        return output

    async def asearch(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        client = self._client.get_async_search_client(collection)
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=limit,
            fields=self._vector_field,
        )
        results = await client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=self._build_filter(filters),
            top=limit,
        )
        output: list[VectorSearchResult] = []
        async for doc in results:
            score: float = float(doc.get("@search.score", 0.0))
            if score_threshold is not None and score < score_threshold:
                continue
            output.append(self._document_to_search_result(doc, score))
        return output

    # Get by ID

    def get(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool = False,
    ) -> list[VectorRecord]:
        client = self._client.get_search_client(collection)
        records: list[VectorRecord] = []
        for doc_id in ids:
            try:
                doc = client.get_document(key=doc_id)
                records.append(self._document_to_record(doc, with_vectors))
            except ResourceNotFoundError:
                continue
        return records

    async def aget(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool = False,
    ) -> list[VectorRecord]:
        client = self._client.get_async_search_client(collection)
        records: list[VectorRecord] = []
        for doc_id in ids:
            try:
                doc = await client.get_document(key=doc_id)
                records.append(self._document_to_record(doc, with_vectors))
            except ResourceNotFoundError:
                continue
        return records

    # Delete

    def delete(self, collection: str, ids: list[str]) -> None:
        client = self._client.get_search_client(collection)
        documents = [{self._key_field: doc_id} for doc_id in ids]
        for i in range(0, len(documents), VECTOR_UPSERT_BATCH_SIZE):
            client.delete_documents(documents[i : i + VECTOR_UPSERT_BATCH_SIZE])

    async def adelete(self, collection: str, ids: list[str]) -> None:
        client = self._client.get_async_search_client(collection)
        documents = [{self._key_field: doc_id} for doc_id in ids]
        for i in range(0, len(documents), VECTOR_UPSERT_BATCH_SIZE):
            await client.delete_documents(documents[i : i + VECTOR_UPSERT_BATCH_SIZE])

    # Count

    def count(self, collection: str) -> int:
        client = self._client.get_search_client(collection)
        return client.get_document_count()

    async def acount(self, collection: str) -> int:
        client = self._client.get_async_search_client(collection)
        return await client.get_document_count()

    # Scroll

    def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        client = self._client.get_search_client(collection)
        skip: int = int(offset) if offset else 0
        results = client.search(
            search_text="*",
            filter=self._build_filter(filters),
            top=limit,
            skip=skip,
            include_total_count=True,
        )

        records: list[VectorRecord] = []
        for doc in results:
            records.append(self._document_to_record(doc, with_vectors))

        next_skip: int = skip + len(records)
        total: int | None = getattr(results, "get_count", lambda: None)()
        has_more: bool = total is not None and next_skip < total
        next_offset: str | None = str(next_skip) if has_more else None

        return records, next_offset

    async def ascroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        client = self._client.get_async_search_client(collection)
        skip: int = int(offset) if offset else 0
        results = await client.search(
            search_text="*",
            filter=self._build_filter(filters),
            top=limit,
            skip=skip,
            include_total_count=True,
        )

        records: list[VectorRecord] = []
        async for doc in results:
            records.append(self._document_to_record(doc, with_vectors))

        next_skip: int = skip + len(records)
        total: int | None = await results.get_count()
        has_more: bool = total is not None and next_skip < total
        next_offset: str | None = str(next_skip) if has_more else None

        return records, next_offset

    # Health Check

    def health_check(self) -> bool:
        try:
            list(self._client.sync_index_client.list_index_names())
            return True
        except AzureError as e:
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed: {e}"
            ) from e

    async def ahealth_check(self) -> bool:
        try:
            async for _ in self._client.async_index_client.list_index_names():
                break
            return True
        except AzureError as e:
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed: {e}"
            ) from e

    # Metadata & Lifecycle

    def get_service_name(self) -> str:
        return "Azure AI Search"

    def dispose(self) -> None:
        self._client.close()

    async def adispose(self) -> None:
        await self._client.aclose()

    # Internals

    def _build_index(
        self,
        index_name: str,
        vector_size: int,
        distance: str,
        **kwargs: Any,
    ) -> SearchIndex:
        """Build a ``SearchIndex`` definition with id, vector, and metadata fields."""
        metric: VectorSearchAlgorithmMetric = _METRIC_MAP.get(
            distance, VectorSearchAlgorithmMetric.COSINE
        )
        algorithm_name: str = "hnsw-algorithm"
        profile_name: str = "vector-profile"

        extra_fields: list[SearchField] = kwargs.get("extra_fields", [])

        fields: list[SearchField] = [
            SimpleField(
                name=self._key_field,
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchField(
                name=self._vector_field,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_size,
                vector_search_profile_name=profile_name,
            ),
            SimpleField(
                name="metadata",
                type=SearchFieldDataType.String,
                filterable=False,
            ),
            *extra_fields,
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=algorithm_name,
                    parameters=HnswParameters(metric=metric),
                ),
            ],
            profiles=[
                VectorSearchProfile(
                    name=profile_name,
                    algorithm_configuration_name=algorithm_name,
                ),
            ],
        )

        return SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
        )

    def _record_to_document(self, record: VectorRecord) -> dict[str, Any]:
        """Convert a ``VectorRecord`` to an Azure AI Search document."""
        return {
            self._key_field: record.id,
            self._vector_field: record.vector,
            "metadata": json.dumps(record.metadata) if record.metadata else "{}",
        }

    def _document_to_record(
        self, doc: dict[str, Any], with_vectors: bool = False
    ) -> VectorRecord:
        """Convert an Azure AI Search document to a ``VectorRecord``."""
        raw_metadata: Any = doc.get("metadata", "{}")
        try:
            metadata: dict[str, Any] = (
                json.loads(raw_metadata) if isinstance(raw_metadata, str) else {}
            )
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        raw_vector: Any = doc.get(self._vector_field)
        vector: list[float] = (
            list(raw_vector) if with_vectors and isinstance(raw_vector, list) else []
        )

        return VectorRecord(
            id=str(doc.get(self._key_field, "")),
            vector=vector,
            metadata=metadata,
        )

    def _document_to_search_result(
        self, doc: dict[str, Any], score: float
    ) -> VectorSearchResult:
        """Convert an Azure AI Search document to a ``VectorSearchResult``."""
        raw_metadata: Any = doc.get("metadata", "{}")
        try:
            metadata: dict[str, Any] = (
                json.loads(raw_metadata) if isinstance(raw_metadata, str) else {}
            )
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        raw_vector: Any = doc.get(self._vector_field)
        vector: list[float] | None = (
            list(raw_vector) if isinstance(raw_vector, list) else None
        )

        return VectorSearchResult(
            id=str(doc.get(self._key_field, "")),
            score=score,
            vector=vector,
            metadata=metadata,
        )

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> str | None:
        """Convert simple ``{field: value}`` dict to an OData filter string.

        - String values: ``field eq 'value'``
        - Numeric values: ``field eq value``
        - Boolean values: ``field eq true/false``
        - List values: ``(field eq 'a' or field eq 'b')``
        """
        if not filters:
            return None
        parts: list[str] = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions = " or ".join(
                    f"{key} eq '{v}'" if isinstance(v, str) else f"{key} eq {v}"
                    for v in value
                )
                parts.append(f"({conditions})")
            elif isinstance(value, str):
                parts.append(f"{key} eq '{value}'")
            elif isinstance(value, bool):
                parts.append(f"{key} eq {'true' if value else 'false'}")
            elif isinstance(value, (int, float)):
                parts.append(f"{key} eq {value}")
        return " and ".join(parts) if parts else None
