from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar, final

from base.models.definitions.errors import DatabaseErrors
from base.models.view_models.database import FilteredPaginationParams, PaginatedResult
from base.tools.errors import raise_error

T = TypeVar("T")


# Row store


class AbstractStore(ABC, Generic[T]):
    """Database-agnostic abstract store. Works with SQL and NoSQL services.

    Subclasses are responsible for:
    - Setting up their own database service (SQL or NoSQL) in __init__
    - Implementing all abstract CRUD methods
    - Overriding _duplicate_error_types() to map their DB's duplicate-key
      exception (e.g. SQLAlchemy IntegrityError, PyMongo DuplicateKeyError)
    """

    # Duplicate error hook

    def _duplicate_error_types(self) -> tuple[type[BaseException], ...]:
        """Return driver-specific duplicate-key exception types.

        Empty tuple means the store does not translate duplicates — callers
        see the raw driver error.
        """
        return ()

    # Get single

    @abstractmethod
    def _get(self, value: Any, column_name: str = "id") -> T | None:
        """Retrieve a single entity by column value."""

    @abstractmethod
    async def _aget(self, value: Any, column_name: str = "id") -> T | None:
        """Asynchronously retrieve a single entity by column value."""

    @final
    def get(self, value: Any, column_name: str = "id") -> T:
        result = self._get(value, column_name)
        return (
            result
            if result is not None
            else raise_error(DatabaseErrors.RECORD_NOT_FOUND)
        )

    @final
    async def aget(self, value: Any, column_name: str = "id") -> T:
        result = await self._aget(value, column_name)
        return (
            result
            if result is not None
            else raise_error(DatabaseErrors.RECORD_NOT_FOUND)
        )

    # Get many

    @abstractmethod
    def _get_many(self, keys: list[Any], column_name: str = "id") -> list[T]:
        """Retrieve many entities by a list of column values."""

    @abstractmethod
    async def _aget_many(self, keys: list[Any], column_name: str = "id") -> list[T]:
        """Asynchronously retrieve many entities by a list of column values."""

    @final
    def get_many(self, keys: list[Any], column_name: str = "id") -> list[T]:
        return self._get_many(keys, column_name)

    @final
    async def aget_many(self, keys: list[Any], column_name: str = "id") -> list[T]:
        return await self._aget_many(keys, column_name)

    # Add single

    @abstractmethod
    def _add(self, entity: T) -> T:
        """Add a single entity to the database."""

    @abstractmethod
    async def _aadd(self, entity: T) -> T:
        """Asynchronously add a single entity to the database."""

    @final
    def add(self, entity: T) -> T:
        types = self._duplicate_error_types()
        if not types:
            return self._add(entity)
        try:
            return self._add(entity)
        except types:
            raise_error(DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION)

    @final
    async def aadd(self, entity: T) -> T:
        types = self._duplicate_error_types()
        if not types:
            return await self._aadd(entity)
        try:
            return await self._aadd(entity)
        except types:
            raise_error(DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION)

    # Add many

    @abstractmethod
    def _add_many(self, entities: list[T]) -> list[T]:
        """Add many entities to the database."""

    @abstractmethod
    async def _aadd_many(self, entities: list[T]) -> list[T]:
        """Asynchronously add many entities to the database."""

    @final
    def add_many(self, entities: list[T]) -> list[T]:
        types = self._duplicate_error_types()
        if not types:
            return self._add_many(entities)
        try:
            return self._add_many(entities)
        except types:
            raise_error(DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION)

    @final
    async def aadd_many(self, entities: list[T]) -> list[T]:
        types = self._duplicate_error_types()
        if not types:
            return await self._aadd_many(entities)
        try:
            return await self._aadd_many(entities)
        except types:
            raise_error(DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION)

    # Edit

    @abstractmethod
    def _edit(self, value: Any, entity: T, column_name: str = "id") -> T | None:
        """Edit an entity in the database. Returns None if not found."""

    @abstractmethod
    async def _aedit(self, value: Any, entity: T, column_name: str = "id") -> T | None:
        """Asynchronously edit an entity in the database. Returns None if not found."""

    @final
    def edit(self, value: Any, entity: T, column_name: str = "id") -> T | None:
        types = self._duplicate_error_types()
        if not types:
            return self._edit(value, entity, column_name)
        try:
            return self._edit(value, entity, column_name)
        except types:
            raise_error(DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION)

    @final
    async def aedit(self, value: Any, entity: T, column_name: str = "id") -> T | None:
        types = self._duplicate_error_types()
        if not types:
            return await self._aedit(value, entity, column_name)
        try:
            return await self._aedit(value, entity, column_name)
        except types:
            raise_error(DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION)

    # Delete

    @abstractmethod
    def _delete(self, value: Any, column_name: str = "id") -> bool:
        """Soft delete an entity in the database. Returns False if not found."""

    @abstractmethod
    async def _adelete(self, value: Any, column_name: str = "id") -> bool:
        """Asynchronously soft delete an entity in the database. Returns False if not found."""

    @final
    def delete(self, value: Any, column_name: str = "id") -> bool:
        return self._delete(value, column_name)

    @final
    async def adelete(self, value: Any, column_name: str = "id") -> bool:
        return await self._adelete(value, column_name)

    # Get all

    @abstractmethod
    def _get_all(self, **kwargs: Any) -> list[T]:
        """Retrieve all entities."""

    @final
    def get_all(self, **kwargs: Any) -> list[T]:
        return self._get_all(**kwargs)

    @abstractmethod
    async def _aget_all(self, **kwargs: Any) -> list[T]:
        """Asynchronously retrieve all entities."""

    @final
    async def aget_all(self, **kwargs: Any) -> list[T]:
        return await self._aget_all(**kwargs)

    # Get all paged

    @abstractmethod
    def _get_all_paged(
        self, filter_page_params: FilteredPaginationParams, **kwargs: Any
    ) -> PaginatedResult[T]:
        """Retrieve all entities with pagination."""

    @final
    def get_all_paged(
        self, filter_page_params: FilteredPaginationParams, **kwargs: Any
    ) -> PaginatedResult[T]:
        return self._get_all_paged(filter_page_params, **kwargs)

    @abstractmethod
    async def _aget_all_paged(
        self, filter_page_params: FilteredPaginationParams, **kwargs: Any
    ) -> PaginatedResult[T]:
        """Asynchronously retrieve all entities with pagination."""

    @final
    async def aget_all_paged(
        self, filter_page_params: FilteredPaginationParams, **kwargs: Any
    ) -> PaginatedResult[T]:
        """Asynchronously retrieve all entities with pagination."""
        return await self._aget_all_paged(filter_page_params, **kwargs)


# Sequence store


class _HasSequence(Protocol):
    sequence: int


TParent = TypeVar("TParent")
TItem = TypeVar("TItem", bound=_HasSequence)


class AbstractSequenceStore(ABC, Generic[TParent, TItem]):
    """Abstract contract for stores of ordered items scoped under a parent key.

    Sibling to AbstractStore[T]. Use this instead of AbstractStore when the
    data you're storing is:

      - Grouped under a parent entity (conversation, user, device, thread)
      - Ordered by a monotonic per-parent sequence number
      - Appended far more often than edited (typically append-only)
      - Sometimes cleared or range-replaced as a group, but never edited
        item-by-item

    Concrete implementations exist per backend (SQL, Mongo, in-memory) and
    per domain (chat history, event log, notifications). A concrete store
    combines both axes — e.g. SqlChatHistoryStore, MongoChatHistoryStore.

    Type parameters:
      TParent: parent key type (e.g. ConversationId, UserId). Must be
               hashable and serializable by the backing store.
      TItem:   stored item type. Must expose a `sequence: int` attribute.
               Typically a Pydantic view model; backend-specific DTOs are
               an implementation detail of concrete stores.

    Subclasses are responsible for:
      - Setting up their backing store (SQLService, MongoDBService, etc.)
        in __init__.
      - Implementing all abstract _a* async methods and _* sync methods.
    """

    # Append

    @abstractmethod
    def _append(self, parent_id: TParent, items: list[TItem]) -> list[TItem]:
        """Append one or more items to the parent's sequence.

        Items may be passed without storage-assigned fields (e.g. IDs). The
        store populates those fields during persistence and returns the
        resulting items. Concrete implementations must preserve the order of
        `items` in storage and in the return value."""

    @abstractmethod
    async def _aappend(self, parent_id: TParent, items: list[TItem]) -> list[TItem]:
        """Asynchronously append one or more items to the parent's sequence.

        Items may be passed without storage-assigned fields (e.g. IDs). The
        store populates those fields during persistence and returns the
        resulting items. Concrete implementations must preserve the order of
        `items` in storage and in the return value."""

    @final
    def append(self, parent_id: TParent, items: list[TItem]) -> list[TItem]:
        return self._append(parent_id, items)

    @final
    async def aappend(self, parent_id: TParent, items: list[TItem]) -> list[TItem]:
        return await self._aappend(parent_id, items)

    # Get

    @abstractmethod
    def _get(
        self,
        parent_id: TParent,
        *,
        limit: int | None = None,
    ) -> list[TItem]:
        """Return items for a parent in ascending sequence order.

        If limit is None, returns all items. If limit is set, returns the
        most recent `limit` items, still in ascending order. Empty list if
        the parent has no items."""

    @abstractmethod
    async def _aget(
        self,
        parent_id: TParent,
        *,
        limit: int | None = None,
    ) -> list[TItem]:
        """Asynchronously return items for a parent in ascending sequence order.

        If limit is None, returns all items. If limit is set, returns the
        most recent `limit` items, still in ascending order. Empty list if
        the parent has no items."""

    @final
    def get(
        self,
        parent_id: TParent,
        *,
        limit: int | None = None,
    ) -> list[TItem]:
        return self._get(parent_id, limit=limit)

    @final
    async def aget(
        self,
        parent_id: TParent,
        *,
        limit: int | None = None,
    ) -> list[TItem]:
        return await self._aget(parent_id, limit=limit)

    # Count

    @abstractmethod
    def _count(self, parent_id: TParent) -> int:
        """Return the total item count for a parent. Zero if none exist."""

    @abstractmethod
    async def _acount(self, parent_id: TParent) -> int:
        """Asynchronously return the total item count for a parent.
        Zero if none exist."""

    @final
    def count(self, parent_id: TParent) -> int:
        return self._count(parent_id)

    @final
    async def acount(self, parent_id: TParent) -> int:
        return await self._acount(parent_id)

    # Next sequence

    @abstractmethod
    def _next_sequence(self, parent_id: TParent) -> int:
        """Return the next sequence number to assign for this parent.

        Implementations should ensure atomicity under concurrent callers:
        returned values must never be reused and must be strictly greater
        than any previously returned value for the same parent. Sequence
        numbers start at 1."""

    @abstractmethod
    async def _anext_sequence(self, parent_id: TParent) -> int:
        """Asynchronously return the next sequence number for this parent.

        Implementations should ensure atomicity under concurrent callers:
        returned values must never be reused and must be strictly greater
        than any previously returned value for the same parent. Sequence
        numbers start at 1."""

    @final
    def next_sequence(self, parent_id: TParent) -> int:
        return self._next_sequence(parent_id)

    @final
    async def anext_sequence(self, parent_id: TParent) -> int:
        return await self._anext_sequence(parent_id)

    # Clear

    @abstractmethod
    def _clear(self, parent_id: TParent) -> None:
        """Remove all items for a parent. Idempotent: clearing a parent with
        no items is a no-op, not an error."""

    @abstractmethod
    async def _aclear(self, parent_id: TParent) -> None:
        """Asynchronously remove all items for a parent. Idempotent: clearing
        a parent with no items is a no-op, not an error."""

    @final
    def clear(self, parent_id: TParent) -> None:
        self._clear(parent_id)

    @final
    async def aclear(self, parent_id: TParent) -> None:
        await self._aclear(parent_id)

    # Replace range

    @abstractmethod
    def _replace_range(
        self,
        parent_id: TParent,
        *,
        up_to_sequence: int,
        replacements: list[TItem],
    ) -> list[TItem]:
        """Atomically delete all items with sequence <= up_to_sequence and
        insert `replacements` in their place.

        Replacement items may be passed without storage-assigned fields; the
        store populates them during persistence and returns the resulting
        items. Replacement items' sequence numbers must be <= up_to_sequence —
        validated by the public wrapper before this method is called.

        Implementations must be atomic: on any failure, the pre-call state
        must be preserved. Use a transaction (SQL) or a single update
        operation (Mongo) — never two independent writes."""

    @abstractmethod
    async def _areplace_range(
        self,
        parent_id: TParent,
        *,
        up_to_sequence: int,
        replacements: list[TItem],
    ) -> list[TItem]:
        """Asynchronously delete all items with sequence <= up_to_sequence and
        insert `replacements` in their place.

        Replacement items may be passed without storage-assigned fields; the
        store populates them during persistence and returns the resulting
        items. Replacement items' sequence numbers must be <= up_to_sequence —
        validated by the public wrapper before this method is called.

        Implementations must be atomic: on any failure, the pre-call state
        must be preserved. Use a transaction (SQL) or a single update
        operation (Mongo) — never two independent writes."""

    @final
    def replace_range(
        self,
        parent_id: TParent,
        *,
        up_to_sequence: int,
        replacements: list[TItem],
    ) -> list[TItem]:
        self._validate_replacements(up_to_sequence, replacements)
        return self._replace_range(
            parent_id,
            up_to_sequence=up_to_sequence,
            replacements=replacements,
        )

    @final
    async def areplace_range(
        self,
        parent_id: TParent,
        *,
        up_to_sequence: int,
        replacements: list[TItem],
    ) -> list[TItem]:
        self._validate_replacements(up_to_sequence, replacements)
        return await self._areplace_range(
            parent_id,
            up_to_sequence=up_to_sequence,
            replacements=replacements,
        )

    @staticmethod
    def _validate_replacements(
        up_to_sequence: int,
        replacements: list[TItem],
    ) -> None:
        """Defensive check for replace_range callers.

        Rejects empty replacements (use clear() instead) and any replacement
        whose sequence exceeds up_to_sequence (would conflict with retained
        items)."""
        if not replacements:
            raise ValueError(
                "replace_range requires at least one replacement; "
                "use clear() to remove a range without replacement."
            )
        for i, item in enumerate(replacements):
            if item.sequence > up_to_sequence:
                raise ValueError(
                    f"replacements[{i}].sequence ({item.sequence}) exceeds "
                    f"up_to_sequence ({up_to_sequence}); replacements "
                    f"would conflict with retained items."
                )
