"""Fonctions pures du RAG hybride : tokenisation et fusion de classements (RRF)."""
from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "we", "our", "their",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in STOPWORDS]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion : score = somme de 1/(k + rang) sur chaque classement.

    Robuste car il n'utilise que les rangs (les scores BM25 et cosinus ne sont pas comparables).
    Les égalités sont départagées par identifiant pour rester déterministe.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
