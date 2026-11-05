"""Turning text into vectors, without naming a provider."""

import time
from abc import ABC, abstractmethod
from typing import final

from base.models.view_models.embedding import EmbeddingBatch
from base.services.usage import UsageTracker


class AbstractTextEmbeddingService(ABC):
    """One embedding shape, whatever is behind it.

    Batching lives here rather than in each adapter: splitting a list to fit a
    provider's per-request ceiling is the same work whoever answers, and an
    adapter that reimplemented it could silently return vectors out of order.

    Recording lives here for a second reason: it has to happen once per call,
    and an adapter retries inside itself. Doing it above the retry is also what
    stops a new provider from forgetting to do it at all.
    """

    @property
    @abstractmethod
    def model(self) -> str: ...

    @property
    @abstractmethod
    def dimensions(self) -> int | None:
        """Width of the vectors, when the deployment pins one.

        A vector store needs this before the first embedding exists, to create
        the collection.
        """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """How many texts the provider accepts in one request."""

    @abstractmethod
    def get_service_name(self) -> str: ...

    @abstractmethod
    def _embed(self, texts: list[str]) -> EmbeddingBatch:
        """Embed one batch, in order. Never called with an empty list."""

    @abstractmethod
    async def _aembed(self, texts: list[str]) -> EmbeddingBatch:
        """Embed one batch, in order. Never called with an empty list."""

    @final
    def embed_query(self, text: str) -> list[float]:
        """Embed a single piece of text."""
        return self.embed_documents([text])[0]

    @final
    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single piece of text."""
        return (await self.aembed_documents([text]))[0]

    @final
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, returning one vector per input, in order.

        Each batch is one billable request, so each is recorded separately.
        """
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            started = time.perf_counter()
            error: str | None = None
            result: EmbeddingBatch | None = None
            try:
                result = self._embed(batch)
                vectors.extend(result.vectors)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._record(result, started, error)
        return vectors

    @final
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, returning one vector per input, in order.

        Each batch is one billable request, so each is recorded separately.
        """
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            started = time.perf_counter()
            error: str | None = None
            result: EmbeddingBatch | None = None
            try:
                result = await self._aembed(batch)
                vectors.extend(result.vectors)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._record(result, started, error)
        return vectors

    def _record(
        self, result: EmbeddingBatch | None, started: float, error: str | None
    ) -> None:
        """Account for one embedding request, billed or failed."""
        usage = result.usage if result else None
        UsageTracker().record(
            model=self.model,
            label=self.get_service_name(),
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cached_input_tokens=usage.cached_input_tokens if usage else 0,
            cache_write_tokens=usage.cache_write_tokens if usage else 0,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=error,
        )

    def _batches(self, texts: list[str]) -> list[list[str]]:
        size = max(self.batch_size, 1)
        return [texts[i : i + size] for i in range(0, len(texts), size)]
