"""Relecture humaine interactive (terminal) des décisions de screening et d'éligibilité.

File d'attente : décisions automatiques marquées « à revoir » (et, à l'éligibilité, toute exclusion ou tout
« maybe ») qui n'ont pas encore de décision humaine. Chaque réponse est ajoutée à <étape>/human_overrides.csv ;
l'étape doit ensuite être relancée (reprise sans LLM) pour appliquer les décisions.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Callable

from agents.human import append_override, load_overrides
from agents.schemas import Protocol, ScreeningRecord

STAGES = {
    "screening": ("screening", "decisions.jsonl"),
    "eligibility": ("fulltext", "eligibility.jsonl"),
}
CHOICES = "[i]nclude  [e]xclude  [m]aybe  [k]eep automatic  [s]kip  [q]uit"


def _records(run_dir: Path, stage: str) -> list[ScreeningRecord]:
    folder, name = STAGES[stage]
    path = run_dir / folder / name
    if not path.exists():
        raise SystemExit(f"{path} introuvable : lancez d'abord l'étape « {stage} ».")
    done: dict[str, ScreeningRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = ScreeningRecord.model_validate_json(line)
            done[r.paper_id] = r
    return list(done.values())


def review_queue(run_dir: Path, stage: str) -> list[ScreeningRecord]:
    """Décisions à relire qui n'ont pas encore de décision humaine."""
    overrides = load_overrides(run_dir / STAGES[stage][0])
    out = []
    for r in _records(run_dir, stage):
        if r.paper_id in overrides:
            continue
        if r.needs_human_review or (stage == "eligibility" and r.final_decision in ("exclude", "maybe")):
            out.append(r)
    return out


def _papers(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "records_deduplicated.json"
    return {d["id"]: d for d in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}


def describe(r: ScreeningRecord, paper: dict | None) -> str:
    """Fiche affichée au relecteur humain."""
    w = textwrap.TextWrapper(width=100, initial_indent="    ", subsequent_indent="    ")
    lines = [f"{r.paper_id} — {r.title}"]
    if paper:
        lines.append(f"    {paper.get('year') or 'n.d.'} · {paper.get('journal') or ''} · {paper.get('doi') or ''}")
        abstract = (paper.get("abstract") or "(no abstract)")[:900]
        lines += w.wrap("Abstract: " + abstract)
    if r.evidence_level:
        lines.append(f"    Evidence: {r.evidence_level}")
    for name, d in (("A", r.reviewer_a), ("B", r.reviewer_b), ("Arbiter", r.arbiter)):
        if d is not None:
            lines += w.wrap(f"{name}: {d.decision} ({d.criterion}) — {d.rationale[:350]}")
    lines += w.wrap(f"AUTOMATIC DECISION: {r.final_decision} ({r.final_criterion or '-'}, {r.decided_by}) — {r.final_rationale[:300]}")
    return "\n".join(lines)


def interactive_review(
    run_dir: Path, stage: str, protocol: Protocol | None = None,
    ask: Callable[[str], str] = input, out: Callable[[str], None] = print, limit: int | None = None,
) -> dict:
    """Boucle de relecture ; renvoie le nombre de décisions enregistrées par type."""
    queue = review_queue(run_dir, stage)
    if limit:
        queue = queue[:limit]
    papers = _papers(run_dir)
    folder = run_dir / STAGES[stage][0]
    counts = {"include": 0, "exclude": 0, "maybe": 0, "kept": 0, "skipped": 0}
    if protocol is not None:
        crit = [f"I{i}: {c}" for i, c in enumerate(protocol.inclusion_criteria, 1)]
        crit += [f"E{i}: {c}" for i, c in enumerate(protocol.exclusion_criteria, 1)]
        out("Critères du protocole :\n" + "\n".join("  " + textwrap.shorten(c, 110) for c in crit))
    out(f"\n{len(queue)} décision(s) à relire ({stage}). {CHOICES}\n")
    for k, r in enumerate(queue, 1):
        out(f"\n[{k}/{len(queue)}] " + describe(r, papers.get(r.paper_id)))
        while True:
            choice = ask(f"  Décision ? {CHOICES} : ").strip().lower()[:1]
            if choice in ("i", "e", "m", "k", "s", "q"):
                break
            out("  Réponse non reconnue.")
        if choice == "q":
            break
        if choice == "s":
            counts["skipped"] += 1
            continue
        if choice == "k":
            code = (r.final_criterion or "").split("/")[0] if r.final_decision == "exclude" else ""
            append_override(folder, r.paper_id, r.final_decision, "Automatic decision confirmed by human reviewer", code)
            counts["kept"] += 1
            continue
        decision = {"i": "include", "e": "exclude", "m": "maybe"}[choice]
        note = ask("  Note (justification, obligatoire) : ").strip() or "no note"
        code = ask("  Critère d'exclusion (ex. I1, E2 ; Entrée = aucun) : ").strip().upper() if decision == "exclude" else ""
        append_override(folder, r.paper_id, decision, note, code)
        counts[decision] += 1
    out(f"\nDécisions enregistrées dans {folder / 'human_overrides.csv'} : {counts}")
    return counts
