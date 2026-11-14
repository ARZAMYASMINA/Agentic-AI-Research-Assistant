"""Index RAG hybride : dense (Chroma + embeddings Ollama) + sparse (BM25), fusion RRF."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
import ollama
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from rag.fusion import rrf_fuse, tokenize
from tools.paper import Paper

log = logging.getLogger(__name__)

COLLECTION = "papers"
DOC_PREFIX = "search_document: "  # préfixes recommandés pour nomic-embed-text
QUERY_PREFIX = "search_query: "


def _embed_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _doc_text(p: Paper) -> str:
    return f"{p.title}. {p.abstract}".strip()


def embed_texts(texts: list[str], prefix: str, batch: int = 32) -> list[list[float]]:
    client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [prefix + t for t in texts[i : i + batch]]
        try:
            resp = client.embed(model=_embed_model(), input=chunk)
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise SystemExit(
                    f"Modèle d'embeddings '{_embed_model()}' introuvable. Lancez : ollama pull {_embed_model()}"
                )
            raise
        out.extend(resp["embeddings"])
    return out


@dataclass
class Hit:
    paper: Paper
    score: float
    dense_rank: int | None
    sparse_rank: int | None


class HybridIndex:
    def __init__(self, papers: list[Paper], collection) -> None:
        self.papers = {p.id: p for p in papers}
        self.ids = [p.id for p in papers]
        self.bm25 = BM25Okapi([tokenize(_doc_text(p)) or ["_"] for p in papers])
        self.collection = collection

    # -- construction / chargement ------------------------------------------------
    @staticmethod
    def _client(persist_dir: Path):
        return chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))

    @classmethod
    def build(cls, papers: list[Paper], persist_dir: Path) -> "HybridIndex":
        client = cls._client(persist_dir)
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # noqa: BLE001  première construction
            pass
        col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        embeddings = embed_texts([_doc_text(p) for p in papers], DOC_PREFIX)
        step = 500
        for i in range(0, len(papers), step):
            batch = papers[i : i + step]
            col.add(
                ids=[p.id for p in batch],
                embeddings=embeddings[i : i + step],
                documents=[_doc_text(p) for p in batch],
                metadatas=[
                    {"year": p.year or 0, "journal": p.journal, "doi": p.doi, "sources": ",".join(p.sources)}
                    for p in batch
                ],
            )
        return cls(papers, col)

    @classmethod
    def load(cls, run_dir: Path) -> "HybridIndex":
        papers = load_papers(run_dir)
        col = cls._client(run_dir / "chroma").get_collection(COLLECTION)
        return cls(papers, col)

    # -- recherche ---------------------------------------------------------------
    def search(self, query: str, k: int = 10, k_each: int = 50) -> list[Hit]:
        # Sparse (BM25)
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.ids)), key=lambda i: (-scores[i], self.ids[i]))
        sparse = [self.ids[i] for i in order[:k_each] if scores[i] > 0]

        # Dense
        qvec = embed_texts([query], QUERY_PREFIX)[0]
        res = self.collection.query(query_embeddings=[qvec], n_results=min(k_each, len(self.ids)))
        dense = res["ids"][0]

        fused = rrf_fuse([dense, sparse])[:k]
        return [
            Hit(
                paper=self.papers[pid],
                score=score,
                dense_rank=dense.index(pid) + 1 if pid in dense else None,
                sparse_rank=sparse.index(pid) + 1 if pid in sparse else None,
            )
            for pid, score in fused
        ]


def load_papers(run_dir: Path) -> list[Paper]:
    data = json.loads((run_dir / "records_deduplicated.json").read_text(encoding="utf-8"))
    return [Paper.model_validate(d) for d in data]


def build_index(run_dir: Path) -> HybridIndex:
    papers = load_papers(run_dir)
    if not papers:
        raise SystemExit("Aucun article dans records_deduplicated.json : rien à indexer.")
    print(f"  Indexation de {len(papers)} articles (modèle d'embeddings : {_embed_model()})...", flush=True)
    index = HybridIndex.build(papers, run_dir / "chroma")
    (run_dir / "index_info.json").write_text(
        json.dumps({"embedding_model": _embed_model(), "documents": len(papers), "collection": COLLECTION}, indent=2),
        encoding="utf-8",
    )
    return index
