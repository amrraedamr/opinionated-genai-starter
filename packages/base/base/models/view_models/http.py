from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field


class HttpResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    status_code: int | None = None
    data: dict[str, Any] | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    error: Exception | None = None
    error_detail: dict[str, Any] | str | None = None

    @property
    def failed(self) -> bool:
        return not self.success

    @classmethod
    def ok(
        cls,
        status_code: int,
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Self:
        return cls(
            success=True,
            status_code=status_code,
            data=data,
            headers=headers or {},
        )

    @classmethod
    def fail(
        cls,
        error: Exception,
        status_code: int | None = None,
        error_detail: dict[str, Any] | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Self:
        return cls(
            success=False,
            error=error,
            status_code=status_code,
            headers=headers or {},
            error_detail=error_detail,
        )
