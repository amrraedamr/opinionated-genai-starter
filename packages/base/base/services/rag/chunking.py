"""Splitting text into chunks an embedding model can take."""

import re

from pydantic import Field

from base.models.view_models import BaseModel
from base.models.view_models.models import OpenAIModels
from base.models.view_models.rag import Chunk, Document
from base.tools.tokenizer import TokenCounter

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class ChunkingConfig(BaseModel):
    """How a document is cut up."""

    max_tokens: int = Field(default=512, ge=1)
    """Upper bound per chunk. Embedding models reject anything longer."""

    overlap_tokens: int = Field(default=64, ge=0)
    """Tokens repeated from the end of the previous chunk, so a sentence
    split across a boundary is still retrievable from either side."""

    model: OpenAIModels | str = "gpt-4o"
    """Whose tokenizer decides what a token is."""


class TextChunker:
    """Cuts text on the largest natural boundary that fits the token budget.

    Paragraphs first, then sentences, then raw token windows. A chunk is
    never larger than `max_tokens`, so this never produces something the
    embedding model will refuse.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._config: ChunkingConfig = config or ChunkingConfig()
        if self._config.overlap_tokens >= self._config.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self._counter: TokenCounter = TokenCounter(self._config.model)

    def split(self, document: Document) -> list[Chunk]:
        """Cut one document into chunks, carrying its metadata onto each."""
        pieces = self._segments(document.text)
        if not pieces:
            return []

        return [
            Chunk(
                text=text,
                index=index,
                token_count=self._counter.count(text),
                metadata=dict(document.metadata),
            )
            for index, text in enumerate(self._pack(pieces))
        ]

    def _segments(self, text: str) -> list[str]:
        """Break text down until every piece fits the budget on its own."""
        segments: list[str] = []
        for paragraph in _PARAGRAPH.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if self._counter.count(paragraph) <= self._config.max_tokens:
                segments.append(paragraph)
                continue
            segments.extend(self._split_long(paragraph))
        return segments

    def _split_long(self, paragraph: str) -> list[str]:
        """A paragraph over budget: try sentences, then fall back to tokens."""
        segments: list[str] = []
        for sentence in _SENTENCE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if self._counter.count(sentence) <= self._config.max_tokens:
                segments.append(sentence)
            else:
                segments.extend(self._token_windows(sentence))
        return segments

    def _token_windows(self, text: str) -> list[str]:
        """Last resort for text with no usable boundary, such as a long table."""
        tokens = self._counter.encode(text)
        size = self._config.max_tokens
        return [
            self._counter.decode(tokens[start : start + size])
            for start in range(0, len(tokens), size)
        ]

    def _pack(self, segments: list[str]) -> list[str]:
        """Greedily fill chunks, then prepend the overlap from the one before."""
        chunks: list[str] = []
        current: list[str] = []
        budget = 0

        for segment in segments:
            size = self._counter.count(segment)
            if current and budget + size > self._config.max_tokens:
                chunks.append("\n\n".join(current))
                current, budget = [], 0
            current.append(segment)
            budget += size

        if current:
            chunks.append("\n\n".join(current))

        return self._overlap(chunks)

    def _overlap(self, chunks: list[str]) -> list[str]:
        """Prefix each chunk with the tail of its predecessor."""
        if self._config.overlap_tokens == 0 or len(chunks) < 2:
            return chunks

        overlapped = [chunks[0]]
        for previous, chunk in zip(chunks, chunks[1:], strict=False):
            tail = self._counter.decode(
                self._counter.encode(previous)[-self._config.overlap_tokens :]
            )
            overlapped.append(f"{tail}\n\n{chunk}" if tail else chunk)
        return overlapped
