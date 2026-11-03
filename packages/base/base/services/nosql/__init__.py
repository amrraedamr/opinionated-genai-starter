from abc import ABC, abstractmethod
from typing import Any

from base.models.view_models.database import PaginatedResult, PaginationParams


class AbstractNoSQLService(ABC):
    """
    Abstract base class for NoSQL (document-oriented) services with both
    sync and async support.

    SDK-agnostic — concrete implementations (MongoDB, CosmosDB, etc.)
    provide their own CRUD logic against the underlying driver.
    """

    # Metadata

    @abstractmethod
    def get_service_name(self) -> str: ...

    @abstractmethod
    def dispose(self) -> None: ...

    @abstractmethod
    async def adispose(self) -> None: ...

    # Find

    @abstractmethod
    def find_one(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    @abstractmethod
    async def afind_one(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    @abstractmethod
    def find_many(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 0,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def afind_many(
        self,
        collection: str,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        skip: int = 0,
        limit: int = 0,
    ) -> list[dict[str, Any]]: ...

    # Insert

    @abstractmethod
    def insert_one(
        self,
        collection: str,
        document: dict[str, Any],
    ) -> str: ...

    @abstractmethod
    async def ainsert_one(
        self,
        collection: str,
        document: dict[str, Any],
    ) -> str: ...

    @abstractmethod
    def insert_many(
        self,
        collection: str,
        documents: list[dict[str, Any]],
        ordered: bool = True,
    ) -> list[str]: ...

    @abstractmethod
    async def ainsert_many(
        self,
        collection: str,
        documents: list[dict[str, Any]],
        ordered: bool = True,
    ) -> list[str]: ...

    # Update

    @abstractmethod
    def update_one(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int: ...

    @abstractmethod
    async def aupdate_one(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int: ...

    @abstractmethod
    def update_many(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int: ...

    @abstractmethod
    async def aupdate_many(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int: ...

    # Replace

    @abstractmethod
    def replace_one(
        self,
        collection: str,
        filter: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> int: ...

    @abstractmethod
    async def areplace_one(
        self,
        collection: str,
        filter: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> int: ...

    # Delete

    @abstractmethod
    def delete_one(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int: ...

    @abstractmethod
    async def adelete_one(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int: ...

    @abstractmethod
    def delete_many(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int: ...

    @abstractmethod
    async def adelete_many(
        self,
        collection: str,
        filter: dict[str, Any],
    ) -> int: ...

    # Count & Distinct

    @abstractmethod
    def count(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
    ) -> int: ...

    @abstractmethod
    async def acount(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
    ) -> int: ...

    @abstractmethod
    def distinct(
        self,
        collection: str,
        key: str,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]: ...

    @abstractmethod
    async def adistinct(
        self,
        collection: str,
        key: str,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]: ...

    # Aggregation

    @abstractmethod
    def aggregate(
        self,
        collection: str,
        pipeline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def aaggregate(
        self,
        collection: str,
        pipeline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...

    # Pagination

    @abstractmethod
    def paginate(
        self,
        collection: str,
        filter: dict[str, Any],
        page_params: PaginationParams,
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
    ) -> PaginatedResult[dict[str, Any]]: ...

    @abstractmethod
    async def apaginate(
        self,
        collection: str,
        filter: dict[str, Any],
        page_params: PaginationParams,
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
    ) -> PaginatedResult[dict[str, Any]]: ...

    # Health Check

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    async def ahealth_check(self) -> bool: ...
