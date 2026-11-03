from typing import Any, Generic, Self, TypeVar

from humps import camelize
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from pydantic.main import IncEx

T = TypeVar("T")


def to_camel(string: str) -> str:
    return camelize(string)


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )

    def serialize(
        self,
        mode: str = "json",
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        by_alias: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> dict[str, Any]:
        return super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )

    @classmethod
    def deserialize(cls, non_aliased_model_dict: dict) -> Self:
        return cls.model_construct(**non_aliased_model_dict)

    def jsonify(self, by_alias: bool = True) -> str:
        return self.model_dump_json(by_alias=by_alias)


class DebugContext(BaseModel):
    request_id: str
    path: str
    method: str
    traceback: list[str] | None = None


class ErrorResponse(BaseModel, Generic[T]):
    success: bool = False
    error: str
    data: T | None = None
    debug: DebugContext | None = None


class BaseResponse(BaseModel, Generic[T]):
    success: bool = True
    data: T | None = None
