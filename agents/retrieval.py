"""Retrieval Agent : exécute les requêtes sur chaque source, déduplique, produit les chiffres PRISMA."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.schemas import Protocol, SearchStrategy
from tools.dedup import deduplicate
from tools.paper import Paper
from tools.search_apis import search

log = logging.getLogger(__name__)


def run_retrieval(strategy: SearchStrategy, protocol: Protocol, run_dir: Path, max_results: int = 200) -> dict:
    raw_dir = run_dir / "retrieval"
    raw_dir.mkdir(parents=True, exist_ok=True)

    per_source: dict[str, dict] = {}
    errors: dict[str, str] = {}
    all_papers: list[Paper] = []

    for q in strategy.queries:
        print(f"  - {q.source} ...", flush=True)
        try:
            papers, total = search(q.source, q.query, protocol, max_results)
        except Exception as e:  # noqa: BLE001  une source en panne ne bloque pas les autres
            log.error("[%s] échec : %s", q.source, e)
            errors[q.source] = str(e)
            continue
        per_source[q.source] = {"hits_total": total, "retrieved": len(papers)}
        (raw_dir / f"{q.source}.json").write_text(
            json.dumps([p.model_dump() for p in papers], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_papers.extend(papers)
        print(f"    {len(papers)} récupérés (sur {total} résultats)")

    unique, dedup_stats = deduplicate(all_papers)
    (run_dir / "records_deduplicated.json").write_text(
        json.dumps([p.model_dump() for p in unique], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prisma = {
        "identification": {
            "per_source": per_source,
            "total_retrieved": len(all_papers),
            **dedup_stats,
            "without_abstract": sum(1 for p in unique if not p.abstract),
            "max_results_per_source": max_results,
        },
        "errors": errors,
    }
    (run_dir / "prisma_identification.json").write_text(json.dumps(prisma, indent=2), encoding="utf-8")
    return prisma
