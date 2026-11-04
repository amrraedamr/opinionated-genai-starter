"""Turning text into vectors, without naming a provider."""

from abc import ABC, abstractmethod
from typing import final


class AbstractTextEmbeddingService(ABC):
    """One embedding shape, whatever is behind it.

    Batching lives here rather than in each adapter: splitting a list to fit a
    provider's per-request ceiling is the same work whoever answers, and an
    adapter that reimplemented it could silently return vectors out of order.
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
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch, in order. Never called with an empty list."""

    @abstractmethod
    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch, in order. Never called with an empty list."""

    @final
    def embed_query(self, text: str) -> list[float]:
        """Embed a single piece of text."""
        return self._embed([text])[0]

    @final
    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single piece of text."""
        return (await self._aembed([text]))[0]

    @final
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, returning one vector per input, in order."""
        results: list[list[float]] = []
        for batch in self._batches(texts):
            results.extend(self._embed(batch))
        return results

    @final
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, returning one vector per input, in order."""
        results: list[list[float]] = []
        for batch in self._batches(texts):
            results.extend(await self._aembed(batch))
        return results

    def _batches(self, texts: list[str]) -> list[list[str]]:
        size = max(self.batch_size, 1)
        return [texts[i : i + size] for i in range(0, len(texts), size)]
