"""Page arithmetic shared by every paginated backend."""

import math
from collections.abc import Awaitable, Callable
from typing import TypeVar

from base.models.view_models.database import PaginatedResult, PaginationParams

T = TypeVar("T")


def _window(total: int, params: PaginationParams) -> tuple[int, int] | None:
    """Offset and limit for the requested page, or None when it holds nothing."""
    total_pages = math.ceil(total / params.size) if params.size > 0 else 0
    if total == 0 or params.page > total_pages:
        return None
    return (params.page - 1) * params.size, params.size


def _result(data: list[T], total: int, params: PaginationParams) -> PaginatedResult[T]:
    return PaginatedResult(
        data=data,
        page=params.page,
        size=params.size,
        total=total,
        total_pages=math.ceil(total / params.size) if params.size > 0 else 0,
    )


def paginate(
    total: int,
    params: PaginationParams,
    fetch: Callable[[int, int], list[T]],
) -> PaginatedResult[T]:
    """Build a page of results, calling `fetch(skip, limit)` only when one exists."""
    window = _window(total, params)
    data = [] if window is None else fetch(*window)
    return _result(data, total, params)


async def apaginate(
    total: int,
    params: PaginationParams,
    fetch: Callable[[int, int], Awaitable[list[T]]],
) -> PaginatedResult[T]:
    """Build a page of results, awaiting `fetch(skip, limit)` only when one exists."""
    window = _window(total, params)
    data = [] if window is None else await fetch(*window)
    return _result(data, total, params)
