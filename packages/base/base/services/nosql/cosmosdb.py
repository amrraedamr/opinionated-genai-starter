from typing import Any

from base.tools.errors import MissingExtraError

try:
    from azure.cosmos import ContainerProxy, exceptions
    from azure.cosmos.aio import ContainerProxy as AsyncContainerProxy
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("azure", "azure-cosmos", feature="CosmosDBService") from exc

from base.clients.nosql.cosmosdb import CosmosDBClient
from base.models.view_models.database import PaginatedResult, PaginationParams
from base.services.common.meta import SingletonABC
from base.services.configuration import get_settings
from base.services.logging.console import ConsoleLoggerService
from base.services.nosql import AbstractNoSQLService
from base.tools import pagination
from base.tools.errors import ServiceHealthCheckError


class CosmosDBService(AbstractNoSQLService, metaclass=SingletonABC):
    """Azure Cosmos DB (SQL API) service with both sync and async support."""

    def __init__(self) -> None:
        self._client: CosmosDBClient = CosmosDBClient()
        self._database_name: str = get_settings().azure_cosmos_database_name
        self._logger: ConsoleLoggerService = ConsoleLoggerService()

    def _get_container(self, collection: str) -> ContainerProxy:
        return self._client.get_container(self._database_name, collection)

    def _aget_container(self, collection: str) -> AsyncContainerProxy:
        return self._client.aget_container(self._database_name, collection)

    def get_service_name(self) -> str:
        return "CosmosDB"

    def dispose(self) -> None:
        self._client.close()

    async def adispose(self) -> None:
        await self._client.aclose()

    # Filter translation

    @staticmethod
    def _build_where_clause(
        filter: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert a dict filter to a Cosmos SQL WHERE clause with parameters."""
        if not filter:
            return "", []

        conditions: list[str] = []
        parameters: list[dict[str, Any]] = []

        for i, (key, value) in enumerate(filter.items()):
            param_name = f"@p{i}"
            conditions.append(f"c.{key} = {param_name}")
            parameters.append({"name": param_name, "value": value})

        return " AND ".join(conditions), parameters

    @staticmethod
    def _build_select_clause(
        projection: dict[str, Any] | None,
    ) -> str:
        """Convert a projection dict to a Cosmos SQL SELECT clause."""
        if not projection:
            return "c"

        include_fields = [k for k, v in projection.items() if v]
        if not include_fields:
            return "c"

        return ", ".join(f"c.{field}" for field in include_fields)

    @staticmethod
    def _build_order_clause(
        sort: list[tuple[str, int]] | None,
    ) -> str:
        """Convert sort tuples to a Cosmos SQL ORDER BY clause."""
        if not sort:
            return ""

        parts: list[str] = []
        for field, direction in sort:
            order = "ASC" if direction >= 0 else "DESC"
            parts.append(f"c.{field} {order}")

        return " ORDER BY " + ", ".join(parts)

    # Find

    def find_one(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        container = self._get_container(collection)

        # Fast path: direct read by id
        if "id" in filter and len(filter) == 1:
            try:
                return container.read_item(
                    item=filter["id"], partition_key=filter["id"]
                )
            except exceptions.CosmosResourceNotFoundError:
                return None

        select = self._build_select_clause(projection)
        where, params = self._build_where_clause(filter)
        query = f"SELECT TOP 1 {select} FROM c"
        if where:
            query += f" WHERE {where}"

        items = list(
            container.query_items(
                query=query,
                parameters=params or None,
                enable_cross_partition_query=True,
            )
        )
        return items[0] if items else None

    async def afind_one(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        container = self._aget_container(collection)

        if "id" in filter and len(filter) == 1:
            try:
                return await container.read_item(
                    item=filter["id"], partition_key=filter["id"]
                )
            except exceptions.CosmosResourceNotFoundError:
                return None

        select = self._build_select_clause(projection)
        where, params = self._build_where_clause(filter)
        query = f"SELECT TOP 1 {select} FROM c"
        if where:
            query += f" WHERE {where}"

        items: list[dict[str, Any]] = []
        async for item in container.query_items(
            query=query,
            parameters=params or None,
            enable_cross_partition_query=True,
        ):
            items.append(item)
            break

        return items[0] if items else None

    def find_many(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        container = self._get_container(collection)

        select = self._build_select_clause(projection)
        where, params = self._build_where_clause(filter)
        order = self._build_order_clause(sort)

        query = f"SELECT {select} FROM c"
        if where:
            query += f" WHERE {where}"
        query += order

        if skip:
            query += f" OFFSET {skip}"
            query += f" LIMIT {limit}" if limit else " LIMIT 2147483647"
        elif limit:
            query = query.replace(f"SELECT {select}", f"SELECT TOP {limit} {select}")

        return list(
            container.query_items(
                query=query,
                parameters=params or None,
                enable_cross_partition_query=True,
            )
        )

    async def afind_many(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        container = self._aget_container(collection)

        select = self._build_select_clause(projection)
        where, params = self._build_where_clause(filter)
        order = self._build_order_clause(sort)

        query = f"SELECT {select} FROM c"
        if where:
            query += f" WHERE {where}"
        query += order

        if skip:
            query += f" OFFSET {skip}"
            query += f" LIMIT {limit}" if limit else " LIMIT 2147483647"
        elif limit:
            query = query.replace(f"SELECT {select}", f"SELECT TOP {limit} {select}")

        items: list[dict[str, Any]] = []
        async for item in container.query_items(
            query=query,
            parameters=params or None,
            enable_cross_partition_query=True,
        ):
            items.append(item)

        return items

    # Insert

    def insert_one(
        self,
        collection: str,
        document: dict[str, Any],
    ) -> str:
        container = self._get_container(collection)
        result = container.upsert_item(body=document)
        return str(result["id"])

    async def ainsert_one(
        self,
        collection: str,
        document: dict[str, Any],
    ) -> str:
        container = self._aget_container(collection)
        result = await container.upsert_item(body=document)
        return str(result["id"])

    def insert_many(
        self,
        collection: str,
        documents: list[dict[str, Any]],
        ordered: bool = True,
    ) -> list[str]:
        container = self._get_container(collection)
        ids: list[str] = []
        for doc in documents:
            try:
                result = container.upsert_item(body=doc)
                ids.append(str(result["id"]))
            except exceptions.CosmosHttpResponseError as e:
                if ordered:
                    raise
                self._logger.error(
                    "Failed to insert document in batch for collection %s: %s",
                    collection,
                    e,
                )
        return ids

    async def ainsert_many(
        self,
        collection: str,
        documents: list[dict[str, Any]],
        ordered: bool = True,
    ) -> list[str]:
        container = self._aget_container(collection)
        ids: list[str] = []
        for doc in documents:
            try:
                result = await container.upsert_item(body=doc)
                ids.append(str(result["id"]))
            except exceptions.CosmosHttpResponseError as e:
                if ordered:
                    raise
                self._logger.error(
                    "Failed to insert document in batch for collection %s: %s",
                    collection,
                    e,
                )
        return ids

    # Update

    def update_one(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        document = self.find_one(collection, filter)
        if document is None:
            if not upsert:
                return 0
            document = dict(filter)

        fields = update.get("$set", update)
        document.update(fields)

        container = self._get_container(collection)
        container.upsert_item(body=document)
        return 1

    async def aupdate_one(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        document = await self.afind_one(collection, filter)
        if document is None:
            if not upsert:
                return 0
            document = dict(filter)

        fields = update.get("$set", update)
        document.update(fields)

        container = self._aget_container(collection)
        await container.upsert_item(body=document)
        return 1

    def update_many(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        documents = self.find_many(collection, filter)
        if not documents:
            if upsert:
                self.update_one(collection, filter, update, upsert=True)
                return 1
            return 0

        fields = update.get("$set", update)
        container = self._get_container(collection)
        count = 0
        for doc in documents:
            doc.update(fields)
            container.upsert_item(body=doc)
            count += 1
        return count

    async def aupdate_many(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        documents = await self.afind_many(collection, filter)
        if not documents:
            if upsert:
                await self.aupdate_one(collection, filter, update, upsert=True)
                return 1
            return 0

        fields = update.get("$set", update)
        container = self._aget_container(collection)
        count = 0
        for doc in documents:
            doc.update(fields)
            await container.upsert_item(body=doc)
            count += 1
        return count

    # Replace

    def replace_one(
        self,
        collection: str,
        filter: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        document = self.find_one(collection, filter)
        if document is None:
            if not upsert:
                return 0
            replacement.setdefault("id", filter.get("id"))

        container = self._get_container(collection)
        container.upsert_item(body=replacement)
        return 1

    async def areplace_one(
        self,
        collection: str,
        filter: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        document = await self.afind_one(collection, filter)
        if document is None:
            if not upsert:
                return 0
            replacement.setdefault("id", filter.get("id"))

        container = self._aget_container(collection)
        await container.upsert_item(body=replacement)
        return 1

    # Delete

    def delete_one(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int:
        document = self.find_one(collection, filter)
        if document is None:
            return 0

        container = self._get_container(collection)
        container.delete_item(item=document["id"], partition_key=document["id"])
        return 1

    async def adelete_one(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int:
        document = await self.afind_one(collection, filter)
        if document is None:
            return 0

        container = self._aget_container(collection)
        await container.delete_item(item=document["id"], partition_key=document["id"])
        return 1

    def delete_many(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int:
        documents = self.find_many(collection, filter)
        if not documents:
            return 0

        container = self._get_container(collection)
        count = 0
        for doc in documents:
            container.delete_item(item=doc["id"], partition_key=doc["id"])
            count += 1
        return count

    async def adelete_many(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int:
        documents = await self.afind_many(collection, filter)
        if not documents:
            return 0

        container = self._aget_container(collection)
        count = 0
        for doc in documents:
            await container.delete_item(item=doc["id"], partition_key=doc["id"])
            count += 1
        return count

    # Count & Distinct

    def count(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
    ) -> int:
        container = self._get_container(collection)
        where, params = self._build_where_clause(filter or {})
        query = "SELECT VALUE COUNT(1) FROM c"
        if where:
            query += f" WHERE {where}"

        results: list[Any] = list(
            container.query_items(
                query=query,
                parameters=params or None,
                enable_cross_partition_query=True,
            )
        )
        return int(results[0]) if results else 0

    async def acount(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
    ) -> int:
        container = self._aget_container(collection)
        where, params = self._build_where_clause(filter or {})
        query = "SELECT VALUE COUNT(1) FROM c"
        if where:
            query += f" WHERE {where}"

        results: list[Any] = []
        async for item in container.query_items(
            query=query,
            parameters=params or None,
            enable_cross_partition_query=True,
        ):
            results.append(item)

        return int(results[0]) if results else 0

    def distinct(
        self,
        collection: str,
        key: str,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        container = self._get_container(collection)
        where, params = self._build_where_clause(filter or {})
        query = f"SELECT DISTINCT VALUE c.{key} FROM c"
        if where:
            query += f" WHERE {where}"

        return list(
            container.query_items(
                query=query,
                parameters=params or None,
                enable_cross_partition_query=True,
            )
        )

    async def adistinct(
        self,
        collection: str,
        key: str,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        container = self._aget_container(collection)
        where, params = self._build_where_clause(filter or {})
        query = f"SELECT DISTINCT VALUE c.{key} FROM c"
        if where:
            query += f" WHERE {where}"

        items: list[Any] = []
        async for item in container.query_items(
            query=query,
            parameters=params or None,
            enable_cross_partition_query=True,
        ):
            items.append(item)

        return items

    # Aggregation

    def aggregate(
        self,
        collection: str,
        pipeline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run Cosmos SQL queries as aggregation stages.

        Each stage is a dict with 'query' and optional 'parameters' keys.
        e.g. [{"query": "SELECT c.status, COUNT(1) AS cnt FROM c GROUP BY c.status"}]
        """
        container = self._get_container(collection)

        if not pipeline:
            return []

        results: list[dict[str, Any]] = []
        for stage in pipeline:
            query = stage.get("query", "")
            params = stage.get("parameters")
            items = list(
                container.query_items(
                    query=query,
                    parameters=params,
                    enable_cross_partition_query=True,
                )
            )
            results.extend(items)
        return results

    async def aaggregate(
        self,
        collection: str,
        pipeline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run Cosmos SQL queries as aggregation stages asynchronously."""
        container = self._aget_container(collection)

        if not pipeline:
            return []

        results: list[dict[str, Any]] = []
        for stage in pipeline:
            query = stage.get("query", "")
            params = stage.get("parameters")
            async for item in container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True,
            ):
                results.append(item)
        return results

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

    # Health Check

    def health_check(self) -> bool:
        if not self._client.ping():
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed"
            )
        return True

    async def ahealth_check(self) -> bool:
        if not await self._client.aping():
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed"
            )
        return True
