from enum import Enum
from typing import Never


class ErrorsException(Exception):
    def __init__(
        self,
        message: str | None = None,
        status_code: int | None = None,
        error_enum: Enum | None = None,
    ) -> None:
        self.error_enum = error_enum
        self.message = message
        self.status_code = status_code
        super().__init__(
            f"Error: {error_enum.value['message']}"
            if error_enum
            else message or "Unknown Error"
        )


class StateDecodeError(ValueError):
    """Raised when the encoded state is invalid, tampered, or cannot be decoded."""


class MissingExtraError(ImportError):
    """Raised when a subsystem is used without the extra that provides it."""

    def __init__(
        self, extra: str, distribution: str, feature: str | None = None
    ) -> None:
        self.extra = extra
        self.distribution = distribution
        subject = feature or f"The {distribution!r} integration"
        super().__init__(
            f'{subject} requires the {extra!r} extra. Add "base[{extra}]" to '
            f"this service's pyproject.toml (installs {distribution})."
        )


class ServiceHealthCheckError(Exception):
    """Raised when a backend service health probe fails."""


def raise_error(
    error: Enum | None = None,
    message: str | None = None,
    status_code: int | None = None,
) -> Never:
    if error is not None:
        raise ErrorsException(error_enum=error)

    raise ErrorsException(message=message, status_code=status_code)
