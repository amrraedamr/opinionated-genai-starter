"""`TextChunker` cuts on the largest natural boundary that fits the budget.

What matters downstream is that no chunk is bigger than the embedding model
will take, that a document's metadata survives the cut, and that the overlap
is a prefix taken from the previous chunk rather than a second copy of it.
"""

from __future__ import annotations

import pytest

from base.models.view_models.rag import ChunkingConfig, Document
from base.services.rag.chunking import TextChunker
from base.tools.tokenizer import TokenCounter

COUNTER = TokenCounter(model="gpt-4o")


def chunker(max_tokens: int = 32, overlap_tokens: int = 0) -> TextChunker:
    return TextChunker(
        ChunkingConfig(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    )


def document(text: str, **metadata: object) -> Document:
    return Document(id="doc", text=text, metadata=dict(metadata))


def test_an_overlap_at_least_as_large_as_a_chunk_is_refused() -> None:
    with pytest.raises(ValueError, match="overlap_tokens must be smaller"):
        TextChunker(ChunkingConfig(max_tokens=10, overlap_tokens=10))


def test_an_empty_document_produces_no_chunks() -> None:
    assert chunker().split(document("")) == []


def test_a_whitespace_only_document_produces_no_chunks() -> None:
    assert chunker().split(document("   \n\n  \t ")) == []


def test_a_document_shorter_than_one_chunk_is_a_single_chunk() -> None:
    text = "A short note."
    chunks = chunker(max_tokens=100).split(document(text))

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].index == 0
    assert chunks[0].token_count == COUNTER.count(text)


def test_chunks_are_indexed_from_zero_in_order() -> None:
    text = "\n\n".join(f"Paragraph number {n} with a little text." for n in range(6))
    chunks = chunker(max_tokens=20).split(document(text))

    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_no_chunk_exceeds_the_token_budget() -> None:
    text = "\n\n".join(f"Paragraph {n} " + "word " * 20 for n in range(8))
    chunks = chunker(max_tokens=40).split(document(text))

    assert chunks
    assert all(c.token_count <= 40 for c in chunks)


def test_a_paragraph_over_budget_is_cut_on_sentences() -> None:
    sentence = "This sentence is roughly ten tokens long indeed."
    chunks = chunker(max_tokens=16).split(document(" ".join([sentence] * 4)))

    assert len(chunks) == 4
    assert all(c.text == sentence for c in chunks)


def test_text_with_no_boundary_at_all_is_cut_on_raw_token_windows() -> None:
    text = "x" * 400
    chunks = chunker(max_tokens=8).split(document(text))

    assert len(chunks) > 1
    assert all(c.token_count <= 8 for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_chunks_reassemble_the_paragraphs_when_there_is_no_overlap() -> None:
    paragraphs = [f"Paragraph {n} carries a sentence worth keeping." for n in range(5)]
    chunks = chunker(max_tokens=24, overlap_tokens=0).split(
        document("\n\n".join(paragraphs))
    )

    assert "\n\n".join(c.text for c in chunks) == "\n\n".join(paragraphs)


def test_overlap_prefixes_each_chunk_with_the_tail_of_the_one_before() -> None:
    paragraphs = [f"Paragraph {n} carries a sentence worth keeping." for n in range(4)]
    plain = chunker(max_tokens=24, overlap_tokens=0).split(
        document("\n\n".join(paragraphs))
    )
    overlapped = chunker(max_tokens=24, overlap_tokens=5).split(
        document("\n\n".join(paragraphs))
    )

    assert len(overlapped) == len(plain) > 1
    assert overlapped[0].text == plain[0].text
    for previous, current, bare in zip(plain, overlapped[1:], plain[1:], strict=False):
        tail = COUNTER.decode(COUNTER.encode(previous.text)[-5:])
        assert current.text == f"{tail}\n\n{bare.text}"


def test_a_single_chunk_is_never_given_an_overlap() -> None:
    chunks = chunker(max_tokens=100, overlap_tokens=10).split(document("Short."))
    assert [c.text for c in chunks] == ["Short."]


def test_document_metadata_travels_onto_every_chunk() -> None:
    text = "\n\n".join(f"Paragraph {n} with some text in it." for n in range(4))
    chunks = chunker(max_tokens=16).split(document(text, source="handbook", page=3))

    assert len(chunks) > 1
    assert all(c.metadata == {"source": "handbook", "page": 3} for c in chunks)


def test_each_chunk_gets_its_own_copy_of_the_metadata() -> None:
    text = "\n\n".join(f"Paragraph {n} with some text in it." for n in range(4))
    source = document(text, source="handbook")
    chunks = chunker(max_tokens=16).split(source)

    chunks[0].metadata["source"] = "edited"
    assert source.metadata["source"] == "handbook"
    assert chunks[1].metadata["source"] == "handbook"


def test_blank_lines_between_paragraphs_are_normalised_away() -> None:
    chunks = chunker(max_tokens=200).split(document("First.\n   \n\nSecond."))
    assert [c.text for c in chunks] == ["First.\n\nSecond."]


def test_unicode_text_survives_chunking_intact() -> None:
    paragraphs = [
        "مرحبا بالعالم، هذه فقرة كاملة.",
        "第二段落、ここにあります。",
        "Третий абзац.",
    ]
    chunks = chunker(max_tokens=200).split(document("\n\n".join(paragraphs)))

    assert [c.text for c in chunks] == ["\n\n".join(paragraphs)]


def test_unicode_paragraphs_each_become_their_own_chunk_under_pressure() -> None:
    paragraphs = ["مرحبا بالعالم، هذه فقرة كاملة.", "第二段落、ここにあります。"]
    chunks = chunker(max_tokens=12).split(document("\n\n".join(paragraphs)))

    assert len(chunks) == 2
    assert chunks[0].text == paragraphs[0]
    assert chunks[1].text == paragraphs[1]


def test_a_chunk_carries_the_token_count_of_its_own_text() -> None:
    text = "\n\n".join(f"Paragraph {n} with some text in it." for n in range(4))
    chunks = chunker(max_tokens=16).split(document(text))

    assert all(c.token_count == COUNTER.count(c.text) for c in chunks)


def test_a_chunks_vector_id_is_derived_from_its_document_and_position() -> None:
    chunks = chunker(max_tokens=100).split(document("Only one chunk here."))
    assert chunks[0].vector_id("handbook") == "handbook:0"


def test_the_default_configuration_is_used_when_none_is_given() -> None:
    assert TextChunker()._config == ChunkingConfig()
    assert ChunkingConfig().max_tokens == 512
    assert ChunkingConfig().overlap_tokens == 64
