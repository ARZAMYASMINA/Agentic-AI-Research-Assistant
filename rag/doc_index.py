"""Index hybride en mémoire pour UN document : passages d'un texte intégral (BM25 + embeddings + RRF)."""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunks import chunk_text
from rag.fusion import rrf_fuse, tokenize
from rag.index import DOC_PREFIX, QUERY_PREFIX, embed_texts


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class DocIndex:
    def __init__(self, text: str, max_chars: int = 1500) -> None:
        self.chunks = chunk_text(text, max_chars=max_chars)
        if not self.chunks:
            raise ValueError("Texte vide : rien à indexer")
        self.bm25 = BM25Okapi([tokenize(c) or ["_"] for c in self.chunks])
        self.emb = _normalize(np.array(embed_texts(self.chunks, DOC_PREFIX), dtype=float))

    def search_many(self, queries: list[str], k: int = 2) -> list[list[int]]:
        """Pour chaque requête : indices des k meilleurs passages (fusion dense + BM25)."""
        qvecs = _normalize(np.array(embed_texts(queries, QUERY_PREFIX), dtype=float))
        sims = qvecs @ self.emb.T
        results: list[list[int]] = []
        for qi, query in enumerate(queries):
            dense = [str(i) for i in np.argsort(-sims[qi])[:20]]
            scores = self.bm25.get_scores(tokenize(query))
            sparse = [str(i) for i in np.argsort(-scores)[:20] if scores[i] > 0]
            results.append([int(i) for i, _ in rrf_fuse([dense, sparse])[:k]])
        return results

    def context(self, queries: list[str], per_query: int = 2, max_chars: int = 9000) -> str:
        """Passages les plus pertinents pour l'ensemble des requêtes, remis dans l'ordre du document."""
        ordered: list[int] = []
        for hits in self.search_many(queries, per_query):
            for i in hits:
                if i not in ordered:
                    ordered.append(i)
        kept, total = [], 0
        for i in ordered:  # par pertinence : si la limite est atteinte, on perd les moins utiles
            if total + len(self.chunks[i]) > max_chars:
                continue
            kept.append(i)
            total += len(self.chunks[i])
        return "\n\n".join(f"[excerpt {i + 1}/{len(self.chunks)}]\n{self.chunks[i]}" for i in sorted(kept))
