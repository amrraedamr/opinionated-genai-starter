from datetime import UTC, datetime
from typing import Annotated, Generic, TypeVar

from pydantic import Field

from base.models.definitions.database import SortingOrder
from base.models.view_models import BaseModel

_DEFAULT_FROM_DATE: datetime = datetime.min.replace(tzinfo=UTC)
_DEFAULT_TO_DATE: datetime = datetime.max.replace(tzinfo=UTC)

T = TypeVar("T")


class PaginatedResult(BaseModel, Generic[T]):
    data: list[T]
    page: int
    size: int
    total: int
    total_pages: int


class PaginationParams(BaseModel):
    """Request query params for paginated API."""

    page: Annotated[int, Field(ge=1)] = 1
    size: Annotated[int, Field(ge=1, le=100)] = 10
    search: str = ""


class SortingParams(BaseModel):
    sort_by: str | None = None
    sort_order: SortingOrder | None = None


class PagedResponse(BaseModel):
    """Response schema for any paged API."""

    total_records: int
    total_pages: int
    page_number: int
    page_size: int


class PagedDataResponse(BaseModel, Generic[T]):
    """Response schema for any paged API with data."""

    data: list[T]
    pagination: PagedResponse


class FilteredPaginationParams(PaginationParams):
    """Request query params for API with pagination and date filtering."""

    from_date: datetime = Field(default=_DEFAULT_FROM_DATE)
    to_date: datetime = Field(default=_DEFAULT_TO_DATE)
