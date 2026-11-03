from collections.abc import Sequence
from typing import Any, TypeVar

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy.engine import Result, Row
    from sqlalchemy.engine.row import RowMapping
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="SQL helpers") from exc

_T = TypeVar("_T")


class SQLResultFormats:
    @staticmethod
    def one_or_none(query_result: Result[Any]) -> Row[Any] | None:
        return query_result.first()

    @staticmethod
    def one_or_none_scalar(query_result: Result[tuple[_T]]) -> _T | None:
        return query_result.scalar_one_or_none()

    @staticmethod
    def all_scalar_sequence(query_result: Result[tuple[_T]]) -> Sequence[_T]:
        return query_result.scalars().all()

    @staticmethod
    def all_scalar_list(query_result: Result[tuple[_T]]) -> list[_T]:
        return list(query_result.scalars().all())

    @staticmethod
    def all(query_result: Result[Any]) -> list[Row[Any]]:
        return list(query_result.all())

    @staticmethod
    def all_mappings(query_result: Result[Any]) -> list[RowMapping]:
        return list(query_result.mappings().all())
