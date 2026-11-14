"""Full-text Retrieval : textes intégraux open access des articles retenus au screening titre/résumé.

Écrit dans <run>/fulltext/ : text/<id>.txt, pdf/<id>.pdf, manifest.json (statut par article).
Reprenable : les articles déjà récupérés ne sont pas retéléchargés.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

from tools.fulltext import MIN_CHARS, fetch_fulltext, pdf_to_text
from tools.paper import Paper

log = logging.getLogger(__name__)


def _manual_pdf(out: Path, p: Paper) -> dict | None:
    """PDF déposé à la main dans fulltext/pdf/<id>.pdf (accès institutionnel, par exemple)."""
    pdf = out / "pdf" / f"{p.id}.pdf"
    if not pdf.exists() or (out / "text" / f"{p.id}.txt").exists():
        return None
    text = pdf_to_text(pdf.read_bytes())
    if len(text) < MIN_CHARS:
        log.warning("[%s] PDF manuel illisible ou trop court (%d caractères).", p.id, len(text))
        return None
    return {"method": "manual_pdf", "url": str(pdf), "text": text, "pdf": None}


def run_fulltext_retrieval(run_dir: Path) -> dict:
    src = run_dir / "screening" / "included_for_fulltext.json"
    if not src.exists():
        raise SystemExit("screening/included_for_fulltext.json introuvable : lancez d'abord le screening.")
    papers = [Paper.model_validate(d) for d in json.loads(src.read_text(encoding="utf-8"))]

    out = run_dir / "fulltext"
    (out / "text").mkdir(parents=True, exist_ok=True)
    (out / "pdf").mkdir(exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = {m["id"]: m for m in json.loads(manifest_path.read_text(encoding="utf-8"))}

    for i, p in enumerate(papers, 1):
        prev = manifest.get(p.id)
        if prev and prev["status"] == "full_text" and (out / "text" / f"{p.id}.txt").exists():
            continue
        try:
            res = _manual_pdf(out, p) or fetch_fulltext(p)
        except Exception as e:  # noqa: BLE001  un article problématique ne bloque pas les autres
            log.error("[%s] récupération échouée : %s", p.id, e)
            res = None

        entry = {"id": p.id, "title": p.title, "year": p.year, "doi": p.doi, "pmid": p.pmid}
        if res:
            (out / "text" / f"{p.id}.txt").write_text(res["text"], encoding="utf-8")
            if res["pdf"]:
                (out / "pdf" / f"{p.id}.pdf").write_bytes(res["pdf"])
            entry |= {"status": "full_text", "method": res["method"], "url": res["url"], "chars": len(res["text"])}
        else:
            entry |= {"status": "abstract_only" if p.abstract else "unavailable", "method": "", "url": "", "chars": 0}
        manifest[p.id] = entry
        manifest_path.write_text(json.dumps(list(manifest.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{i}/{len(papers)}] {p.id} {entry['status']:<13} {entry['method'] or '-':<14} {p.title[:60]}", flush=True)
        time.sleep(0.3)

    entries = [manifest[p.id] for p in papers if p.id in manifest]
    summary = {
        "reports_sought": len(papers),
        "full_text": sum(e["status"] == "full_text" for e in entries),
        "abstract_only": sum(e["status"] == "abstract_only" for e in entries),
        "unavailable": sum(e["status"] == "unavailable" for e in entries),
        "by_method": dict(Counter(e["method"] for e in entries if e["status"] == "full_text")),
    }
    (out / "prisma_fulltext_retrieval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_fulltext(s: dict) -> None:
    print("\n=== PRISMA : RÉCUPÉRATION DES TEXTES INTÉGRAUX ===")
    print(f"  Rapports recherchés        : {s['reports_sought']}")
    print(f"  Texte intégral obtenu      : {s['full_text']} {s['by_method'] or ''}")
    print(f"  Résumé seul (pas d'accès libre) : {s['abstract_only']}")
    print(f"  Ni texte ni résumé         : {s['unavailable']}")
