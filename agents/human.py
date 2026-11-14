"""Décisions humaines (human-in-the-loop) pour le screening et l'éligibilité.

Fichier `human_overrides.csv` dans le dossier de l'étape (screening/ ou fulltext/), colonnes :
id, decision (include/exclude/maybe), note, criterion (code I1/E2..., facultatif pour une exclusion).
Les décisions humaines s'appliquent PAR-DESSUS les décisions automatiques, qui restent lisibles dans la
justification finale (piste d'audit). Les fichiers de décisions automatiques ne sont jamais modifiés.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from agents.schemas import ScreeningRecord

log = logging.getLogger(__name__)

OVERRIDES_FILE = "human_overrides.csv"
HEADER = ["id", "decision", "note", "criterion"]


def load_overrides(out_dir: Path) -> dict[str, tuple[str, str, str]]:
    """Lit <out_dir>/human_overrides.csv -> {id: (decision, note, critère)} ; la dernière ligne d'un id l'emporte."""
    path = out_dir / OVERRIDES_FILE
    if not path.exists():
        return {}
    overrides: dict[str, tuple[str, str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("id") or "").strip()
            decision = (row.get("decision") or "").strip().lower()
            if pid and decision in ("include", "exclude", "maybe"):
                overrides[pid] = (decision, (row.get("note") or "").strip(), (row.get("criterion") or "").strip())
            elif pid or decision:
                log.warning("Ligne ignorée dans %s (decision attendue : include/exclude/maybe) : %s", path, dict(row))
    return overrides


def append_override(out_dir: Path, pid: str, decision: str, note: str = "", criterion: str = "") -> None:
    """Ajoute une décision humaine (le fichier est créé avec son en-tête si besoin ; les guillemets CSV sont gérés)."""
    path = out_dir / OVERRIDES_FILE
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow([pid, decision, " ".join(note.split()), criterion])


def apply_override(rec: ScreeningRecord, decision: str, note: str, criterion: str) -> ScreeningRecord:
    """Remplace la décision finale par celle d'un humain ; la décision automatique reste lisible dans la justification."""
    if decision != "exclude":
        crit = "none"
    elif rec.final_criterion == "RULE:not_retrieved":
        crit = rec.final_criterion  # un rapport non récupéré reste compté comme tel dans PRISMA
    else:
        crit = criterion or "human"
    return rec.model_copy(update={
        "final_decision": decision,
        "final_criterion": crit,
        "final_rationale": (
            f"Human decision: {note or 'no note'} "
            f"(automatic decision was: {rec.final_decision} - {rec.final_rationale[:200]})"
        ),
        "decided_by": "human",
        "needs_human_review": False,
    })
