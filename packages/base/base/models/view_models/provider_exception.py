"""What a call to something outside this process can fail with.

A model provider, an embedding endpoint, an object store. The shape of the
failure is the same everywhere — a status code, a request id, whether trying
again could help — so it is described once. Each domain then names its own
family on top, so catching one never catches another.
"""

from typing import Self


class ProviderError(Exception):
    """Base class for every failure of an outbound call."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        provider_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.provider_error = provider_error

    @classmethod
    def from_provider(
        cls,
        error: Exception,
        message: str | None = None,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> Self:
        """Wrap a client SDK exception.

        Pulling `status_code` and `request_id` off the exception is the error
        handler's job, because only it knows the SDK's shape.
        """
        return cls(
            message or str(error),
            status_code=status_code,
            request_id=request_id,
            provider_error=error,
        )

    def __str__(self) -> str:
        parts: list[str] = [super().__str__()]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.request_id is not None:
            parts.append(f"request_id={self.request_id}")
        return " | ".join(parts)


class ProviderConfigError(ProviderError):
    """Invalid configuration. Trying again will not help."""


class ProviderRetryExhausted(ProviderError):
    """A transient failure that outlived its retries.

    Inspect `provider_error` for the cause.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        status_code: int | None = None,
        request_id: str | None = None,
        provider_error: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            request_id=request_id,
            provider_error=provider_error,
        )
        self.attempts = attempts

    def __str__(self) -> str:
        return f"{super().__str__()} | attempts={self.attempts}"
