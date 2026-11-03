import asyncio
import time
from typing import Any

from base.tools.errors import MissingExtraError

try:
    from pymongo.asynchronous.database import AsyncDatabase
    from pymongo.database import Database
    from pymongo.errors import OperationFailure, PyMongoError
    from pymongo.operations import SearchIndexModel
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("mongo", "pymongo", feature="MongoDBService") from exc

from base.clients.nosql.mongodb import MongoDBClient
from base.models.view_models.database import PaginatedResult, PaginationParams
from base.services.common.meta import SingletonABC
from base.services.configuration import get_settings
from base.services.nosql import AbstractNoSQLService
from base.tools import pagination
from base.tools.errors import ServiceHealthCheckError


class MongoDBService(AbstractNoSQLService, metaclass=SingletonABC):
    """MongoDB service with both sync and async support."""

    INDEX_DROP_WAIT_INTERVAL_S: float = 1.0
    INDEX_DROP_WAIT_TIMEOUT_S: float = 60.0

    def __init__(self) -> None:
        super().__init__()
        self._client: MongoDBClient = MongoDBClient()
        self._database_name: str = get_settings().mongodb_database
        self._vector_size: int = get_settings().azure_openai_embedding_dimensions

    def _get_database(self) -> Database[dict[str, Any]]:
        return self._client.get_database(self._database_name)

    def _aget_database(self) -> AsyncDatabase[dict[str, Any]]:
        return self._client.aget_database(self._database_name)

    def get_service_name(self) -> str:
        return "MongoDB"

    def dispose(self) -> None:
        self._client.close()

    async def adispose(self) -> None:
        await self._client.aclose()

    # Find

    def find_one(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return self._get_database()[collection].find_one(filter, projection=projection)

    async def afind_one(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return await self._aget_database()[collection].find_one(
            filter, projection=projection
        )

    def find_many(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        cursor = self._get_database()[collection].find(filter, projection=projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    async def afind_many(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        cursor = self._aget_database()[collection].find(filter, projection=projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=None)

    # Insert

    def insert_one(
        self,
        collection: str,
        document: dict[str, Any],
    ) -> str:
        result = self._get_database()[collection].insert_one(document)
        return str(result.inserted_id)

    async def ainsert_one(
        self,
        collection: str,
        document: dict[str, Any],
    ) -> str:
        result = await self._aget_database()[collection].insert_one(document)
        return str(result.inserted_id)

    def insert_many(
        self,
        collection: str,
        documents: list[dict[str, Any]],
        ordered: bool = True,
    ) -> list[str]:
        result = self._get_database()[collection].insert_many(
            documents, ordered=ordered
        )
        return [str(id) for id in result.inserted_ids]

    async def ainsert_many(
        self,
        collection: str,
        documents: list[dict[str, Any]],
        ordered: bool = True,
    ) -> list[str]:
        result = await self._aget_database()[collection].insert_many(
            documents, ordered=ordered
        )
        return [str(id) for id in result.inserted_ids]

    # Update

    def update_one(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        result = self._get_database()[collection].update_one(
            filter, update, upsert=upsert
        )
        return result.modified_count

    async def aupdate_one(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        result = await self._aget_database()[collection].update_one(
            filter, update, upsert=upsert
        )
        return result.modified_count

    def update_many(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        result = self._get_database()[collection].update_many(
            filter, update, upsert=upsert
        )
        return result.modified_count

    async def aupdate_many(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        result = await self._aget_database()[collection].update_many(
            filter, update, upsert=upsert
        )
        return result.modified_count

    # Replace

    def replace_one(
        self,
        collection: str,
        filter: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        result = self._get_database()[collection].replace_one(
            filter, replacement, upsert=upsert
        )
        return result.modified_count

    async def areplace_one(
        self,
        collection: str,
        filter: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        result = await self._aget_database()[collection].replace_one(
            filter, replacement, upsert=upsert
        )
        return result.modified_count

    # Delete

    def delete_one(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int:
        result = self._get_database()[collection].delete_one(filter)
        return result.deleted_count

    async def adelete_one(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int:
        result = await self._aget_database()[collection].delete_one(filter)
        return result.deleted_count

    def delete_many(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int:
        result = self._get_database()[collection].delete_many(filter)
        return result.deleted_count

    async def adelete_many(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int:
        result = await self._aget_database()[collection].delete_many(filter)
        return result.deleted_count

    # Count & Distinct

    def count(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
    ) -> int:
        return self._get_database()[collection].count_documents(filter or {})

    async def acount(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
    ) -> int:
        return await self._aget_database()[collection].count_documents(filter or {})

    def distinct(
        self,
        collection: str,
        key: str,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        return self._get_database()[collection].distinct(key, filter=filter or {})

    async def adistinct(
        self,
        collection: str,
        key: str,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        return await self._aget_database()[collection].distinct(
            key, filter=filter or {}
        )

    # Aggregation

    def aggregate(
        self,
        collection: str,
        pipeline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return list(self._get_database()[collection].aggregate(pipeline))

    async def aaggregate(
        self,
        collection: str,
        pipeline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        cursor = await self._aget_database()[collection].aggregate(pipeline)
        return await cursor.to_list(length=None)

    # Pagination

    def paginate(
        self,
        collection: str,
        filter: dict[str, Any],
        page_params: PaginationParams,
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
    ) -> PaginatedResult[dict[str, Any]]:
        return pagination.paginate(
            self.count(collection, filter),
            page_params,
            lambda skip, limit: self.find_many(
                collection,
                filter,
                projection=projection,
                sort=sort,
                skip=skip,
                limit=limit,
            ),
        )

    async def apaginate(
        self,
        collection: str,
        filter: dict[str, Any],
        page_params: PaginationParams,
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
    ) -> PaginatedResult[dict[str, Any]]:
        return await pagination.apaginate(
            await self.acount(collection, filter),
            page_params,
            lambda skip, limit: self.afind_many(
                collection,
                filter,
                projection=projection,
                sort=sort,
                skip=skip,
                limit=limit,
            ),
        )

    def create_collection_vector_index(
        self,
        collection: str,
        filter_fields: list[dict[str, str]] | None = None,
    ) -> None:
        """Idempotently ensure the Atlas Search vector index exists with
        the given definition.

        Strategy:
          - If the index does NOT exist → ``create_search_index``.
          - If it DOES exist → try ``update_search_index`` first. Atlas
            allows updating field definitions in place when the change is
            compatible.
          - If ``update_search_index`` fails (e.g. incompatible reshape)
            → ``drop_search_index`` + poll until the drop has propagated
            + ``create_search_index``. This is the only safe way to
            change a fundamentally-shaped index without manual cleanup
            in Atlas UI.
        """
        fields: list[dict[str, Any]] = [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": self._vector_size,
                "similarity": "cosine",
            }
        ]
        for ff in filter_fields or []:
            fields.append(ff)

        index_name = f"{collection}_vector_index"
        definition = {"fields": fields}
        coll = self._get_database()[collection]

        if self._search_index_exists_sync(coll, index_name):
            try:
                coll.update_search_index(index_name, definition)
                return
            except OperationFailure:
                # Fall through to the drop + recreate path.
                pass
            coll.drop_search_index(index_name)
            self._wait_for_search_index_drop_sync(coll, index_name)

        coll.create_search_index(
            SearchIndexModel(
                definition=definition,
                name=index_name,
                type="vectorSearch",
            )
        )

    async def acreate_collection_vector_index(
        self,
        collection: str,
        filter_fields: list[dict[str, str]] | None = None,
    ) -> None:
        """Async sibling of :meth:`create_collection_vector_index`."""
        fields: list[dict[str, Any]] = [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": self._vector_size,
                "similarity": "cosine",
            }
        ]
        for ff in filter_fields or []:
            fields.append(ff)

        index_name = f"{collection}_vector_index"
        definition = {"fields": fields}
        coll = self._aget_database()[collection]

        if await self._asearch_index_exists(coll, index_name):
            try:
                await coll.update_search_index(index_name, definition)
                return
            except OperationFailure:
                pass
            await coll.drop_search_index(index_name)
            await self._await_search_index_drop(coll, index_name)

        await coll.create_search_index(
            SearchIndexModel(
                definition=definition,
                name=index_name,
                type="vectorSearch",
            )
        )

    @staticmethod
    def _search_index_exists_sync(coll: Any, index_name: str) -> bool:
        try:
            existing = list(coll.list_search_indexes(name=index_name))
        except OperationFailure:
            return False
        return bool(existing)

    @staticmethod
    async def _asearch_index_exists(coll: Any, index_name: str) -> bool:
        try:
            cursor = await coll.list_search_indexes(name=index_name)
            existing = await cursor.to_list(length=1)
        except OperationFailure:
            return False
        return bool(existing)

    @classmethod
    def _wait_for_search_index_drop_sync(
        cls,
        coll: Any,
        index_name: str,
    ) -> None:
        """Block until Atlas reports the search index is gone, or until
        the timeout expires. Atlas drops are asynchronous; recreating
        immediately after a drop request will fail with
        ``IndexAlreadyExists``.
        """
        deadline = time.monotonic() + cls.INDEX_DROP_WAIT_TIMEOUT_S
        while time.monotonic() < deadline:
            if not cls._search_index_exists_sync(coll, index_name):
                return
            time.sleep(cls.INDEX_DROP_WAIT_INTERVAL_S)
        raise OperationFailure(
            f"Timed out waiting for Atlas Search index '{index_name}' "
            f"to be dropped after {cls.INDEX_DROP_WAIT_TIMEOUT_S}s."
        )

    @classmethod
    async def _await_search_index_drop(
        cls,
        coll: Any,
        index_name: str,
    ) -> None:
        deadline = time.monotonic() + cls.INDEX_DROP_WAIT_TIMEOUT_S
        while time.monotonic() < deadline:
            if not await cls._asearch_index_exists(coll, index_name):
                return
            await asyncio.sleep(cls.INDEX_DROP_WAIT_INTERVAL_S)
        raise OperationFailure(
            f"Timed out waiting for Atlas Search index '{index_name}' "
            f"to be dropped after {cls.INDEX_DROP_WAIT_TIMEOUT_S}s."
        )

    # Health Check

    def health_check(self) -> bool:
        try:
            self._get_database().command("ping")
            return True
        except PyMongoError as e:
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed: {e}"
            ) from e

    async def ahealth_check(self) -> bool:
        try:
            await self._aget_database().command("ping")
            return True
        except PyMongoError as e:
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed: {e}"
            ) from e
