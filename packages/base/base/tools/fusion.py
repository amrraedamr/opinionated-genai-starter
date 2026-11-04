"""Fuse several ranked lists into one, without comparing their scores."""

from collections.abc import Sequence

RRF_K: int = 60
"""Damping constant from the original RRF paper. Larger flattens the weight
given to top positions; 60 is the published default every backend uses."""


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = RRF_K,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Merge ranked id lists by position, best first.

    Scores from different retrievers are not comparable — a cosine similarity
    and a BM25 relevance share no scale — so fusion uses rank alone. An id
    contributes `1 / (k + rank)` from each list it appears in, which rewards
    documents several retrievers agree on.

    Lists may be different lengths and need not overlap.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, identifier in enumerate(ranking):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + position + 1)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:limit] if limit is not None else ordered
