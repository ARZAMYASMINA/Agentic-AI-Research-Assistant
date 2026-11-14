"""Synthèse SWiM (Synthesis Without Meta-analysis) par direction des effets.

Pourquoi : un LLM de 9B qui rédige librement la synthèse se trompe d'étude, de groupe ou de sens d'effet,
avec ou sans chiffres. Ici le LLM ne fait qu'une micro-tâche vérifiable : relever, pour UNE étude, les paires
(étiquette du groupe IA, valeur) / (étiquette du comparateur, valeur) telles qu'elles sont écrites dans les données.

Le code vérifie chaque paire, puis calcule la direction et rédige tout :
- les deux valeurs sont sur la ligne de données de l'issue (sensitivity, specificity, auc, other_outcomes) ;
- chaque étiquette figure dans le même segment (séparé par « ; ») que sa valeur, et la valeur est le PREMIER
  nombre qui suit l'étiquette (ce n'est ni une borne d'intervalle de confiance, ni la valeur d'un autre groupe) ;
- les deux valeurs sont distinctes et viennent de segments différents.
La direction (▲ en faveur de l'IA, ▼ en faveur du comparateur, ◆ pas de différence nette) est calculée par le
code selon le sens favorable de l'issue et un seuil relatif prédéfini. Le dénombrement par étude et le test du
signe suivent les recommandations SWiM (Campbell et al., BMJ 2020).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from math import comb
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agents.llm import structured_call
from agents.schemas import EvidenceRecord

log = logging.getLogger(__name__)

OUTCOMES: dict[str, dict] = {
    "sensitivity": {"field": "sensitivity", "higher_is_better": True, "label": "Sensitivity"},
    "specificity": {"field": "specificity", "higher_is_better": True, "label": "Specificity"},
    "auc": {"field": "auc", "higher_is_better": True, "label": "AUC"},
    "cancer_detection_rate": {"field": "other_outcomes", "higher_is_better": True, "label": "Cancer detection rate"},
    "recall_rate": {"field": "other_outcomes", "higher_is_better": False, "label": "Recall rate"},
    "reading_time": {"field": "other_outcomes", "higher_is_better": False, "label": "Reading time"},
}
ROLES = {
    "standalone": "AI alone vs radiologists",
    "assisted": "radiologists with AI vs without AI (reading aid, second reader or combination)",
    "triage": "AI triage workflow vs standard workflow",
}
DEFAULT_THRESHOLD = 0.02  # différence relative < 2 % de la valeur du comparateur : « pas de différence nette »


class Comparison(BaseModel):
    outcome: Literal["sensitivity", "specificity", "auc", "cancer_detection_rate", "recall_rate", "reading_time"]
    ai_role: Literal["standalone", "assisted", "triage"]
    intervention_label: str = Field(description="label of the AI arm, copied from the data (e.g. 'AI', 'with AI')")
    intervention_value: str = Field(description="the number that directly follows that label, copied exactly")
    comparator_label: str = Field(description="label of the comparator arm, copied from the data")
    comparator_value: str = Field(description="the number that directly follows that label, copied exactly")


class StudyComparisons(BaseModel):
    comparisons: list[Comparison]  # obligatoire : un schéma recopié par le modèle ne doit pas valider à vide


EXAMPLE = """Example. Data line:
specificity: unaided: 80.1%; AI-supported: 83.0%
Answer:
{"comparisons": [{"outcome": "specificity", "ai_role": "assisted", "intervention_label": "AI-supported",
"intervention_value": "83.0", "comparator_label": "unaided", "comparator_value": "80.1"}]}
If there is no comparison: {"comparisons": []}"""


SYSTEM_SWIM = """You extract head-to-head comparisons from the data of ONE study for a systematic review
(synthesis by direction of effect). Outcomes: sensitivity, specificity, auc, cancer_detection_rate, recall_rate,
reading_time.
For every comparison between an AI arm and a comparator arm where BOTH values are present in the data:
- intervention_label: the group label of the AI arm, copied exactly as written just before its value in the data
  line of that outcome (e.g. "AI", "AI stand-alone", "AI-supported", "with AI", "AI standalone internal");
- intervention_value: the number that directly follows that label, copied exactly (not a confidence bound);
- comparator_label / comparator_value: the same for the comparator arm (e.g. "radiologists", "unaided",
  "radiologists without AI", "First reader");
- ai_role: "standalone" = AI alone compared with radiologists; "assisted" = radiologists reading with AI (or AI
  combined with radiologists) compared with radiologists without AI; "triage" = AI triage workflow compared with
  the standard workflow.
Compare like with like (same dataset, same outcome). Never compute or round a value. Do not invent a comparator
value that is not written. If there is no valid comparison, return an empty list."""

RETRY = """Your answer failed automatic checks:
{problems}
Return the corrected full list. Copy each label and the number that DIRECTLY follows it from the same data line;
drop any comparison whose two values are not both written in the data."""


# --------------------------------------------------------------------------- données et vérification
_NUM = re.compile(r"\d+(?:[.,·]\d+)?")


def _norm(s: str) -> str:
    s = re.sub(r"(?<=\d)[·∙](?=\d)", ".", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # séparateur de milliers
    return " ".join(s.replace("–", "-").replace("−", "-").split())


def _to_float(s: str) -> float | None:
    m = _NUM.search(_norm(s))
    return float(m.group(0).replace(",", ".").replace("·", ".")) if m else None


def outcome_lines(rec: EvidenceRecord) -> dict[str, str]:
    """Lignes de données utilisables (valeurs étayées ou corrigées par un humain seulement)."""
    out = {}
    for name in ("sensitivity", "specificity", "auc", "other_outcomes", "ai_system", "comparator"):
        fv = rec.extracted.get(name)
        if fv is not None and fv.found and fv.grounded:
            out[name] = _norm(fv.value)
    return out


_GROUP = re.compile(
    r"\b(without|with|unaided|aided|radiologists?|readers?|ai|alone|stand-?alone|combined|combination|"
    r"supported|first|second|double|single|consensus)\b", re.I)


def _group_words(text: str) -> set[str]:
    words = {w.lower().replace("-", "") for w in _GROUP.findall(text)}
    return {w[:-1] if w in ("radiologists", "readers") else w for w in words}


def _locate(label: str, value: str, line: str) -> int | None:
    """Indice du segment (« ; ») où `value` est le 1er nombre qui suit `label`, sinon None.

    De plus, tous les mots désignant un groupe (with, without, unaided, radiologists, AI, first...) écrits avant la
    valeur dans ce segment doivent figurer dans l'étiquette : « AI » n'est pas accepté pour « radiologists without AI ».
    """
    lab = _norm(label).lower().strip(" :")
    val = _to_float(value)
    if not lab or val is None:
        return None
    for k, seg in enumerate(line.split(";")):
        low = seg.lower()
        pos = low.find(lab)
        while pos != -1:
            m = _NUM.search(seg, pos + len(lab))
            if m and abs(float(m.group(0).replace(",", ".")) - val) < 1e-9:
                if _group_words(seg[:m.start()]) <= _group_words(lab):
                    return k
            pos = low.find(lab, pos + 1)
    return None


def validate(c: Comparison, lines: dict[str, str]) -> str:
    """Chaîne vide si la comparaison est vérifiée, sinon la raison du rejet."""
    field = OUTCOMES[c.outcome]["field"]
    line = lines.get(field)
    if not line:
        return f"{c.outcome}: no verified data line '{field}'"
    a = _locate(c.intervention_label, c.intervention_value, line)
    b = _locate(c.comparator_label, c.comparator_value, line)
    if a is None:
        return (f"{c.outcome}: '{c.intervention_value}' is not the number directly after "
                f"'{c.intervention_label}' in the {field} line")
    if b is None:
        return (f"{c.outcome}: '{c.comparator_value}' is not the number directly after "
                f"'{c.comparator_label}' in the {field} line")
    if a == b:
        return f"{c.outcome}: intervention and comparator values come from the same group (same segment)"
    return ""


def direction(c: Comparison, threshold: float = DEFAULT_THRESHOLD) -> str:
    """▲ en faveur de l'IA, ▼ en faveur du comparateur, ◆ pas de différence nette (seuil relatif)."""
    x, y = _to_float(c.intervention_value), _to_float(c.comparator_value)
    if x is None or y is None:
        return "?"
    if y == 0 or abs(x - y) / abs(y) < threshold:
        return "◆"
    better = x > y if OUTCOMES[c.outcome]["higher_is_better"] else x < y
    return "▲" if better else "▼"


# --------------------------------------------------------------------------- appel au LLM (avec cache)
def _ask(paper_id: str, lines: dict[str, str], llm, feedback: str = "") -> StudyComparisons:
    data = "\n".join(f"{k}: {v}" for k, v in lines.items())
    user = f"{EXAMPLE}\n\nStudy {paper_id}. Data lines (the only allowed source):\n{data}"
    if feedback:
        user += "\n\n" + feedback
    return structured_call(StudyComparisons, SYSTEM_SWIM, user, agent="swim", llm=llm)


def _key(c: Comparison) -> tuple:
    return (c.outcome, _to_float(c.intervention_value), _to_float(c.comparator_value))


# --------------------------------------------------------------------------- rôle déduit des étiquettes
_ASSIST = {"with", "supported", "aided", "combined", "combination"}
_NOAI = {"without", "unaided"}
_HUMAN = {"radiologist", "reader", "first", "second", "double", "single", "consensus"}


def arm(label: str) -> str | None:
    """Nature d'un groupe d'après son étiquette : 'ai_assisted', 'ai_alone', 'comparator' ou None (indéterminé)."""
    w = _group_words(label)
    if w & _NOAI:
        return "comparator"
    if w & _ASSIST:
        return "ai_assisted"
    if "ai" in w or "alone" in w or "standalone" in w or re.match(r"\s*(dl|ai)[- ]?\d", label, re.I):
        return "ai_alone"
    if w & _HUMAN:
        return "comparator"
    return None


def role_from_labels(c: Comparison) -> str:
    """Le rôle de l'IA est déduit des étiquettes quand c'est possible (le modèle se trompait : IA seule -> « assisted »)."""
    a, b = arm(c.intervention_label), arm(c.comparator_label)
    if a == "ai_assisted":
        return "assisted"
    if a == "ai_alone" and b == "comparator":
        return "standalone"
    return c.ai_role


def hints(lines: dict[str, str]) -> list[str]:
    """Lignes où le code voit au moins un groupe IA et un groupe comparateur étiquetés, chacun suivi d'une valeur."""
    out = []
    for o in ("sensitivity", "specificity", "auc"):
        line = lines.get(o)
        if not line:
            continue
        arms = set()
        for seg in line.split(";"):
            m = _NUM.search(seg)
            if m:
                arms.add(arm(seg[:m.start()]))
        if "comparator" in arms and arms & {"ai_alone", "ai_assisted"}:
            out.append(o)
    return out


def extract_comparisons(records: list[EvidenceRecord], cache_file: Path, llm_factory) -> dict[str, dict]:
    """{paper_id: {"valid": [...], "rejected": [...]}}.

    Un seul nouvel essai, si des paires sont rejetées OU si le code voit des comparaisons possibles (`hints`) pour
    une issue qui n'a aucune paire valide. Le rôle de l'IA est ensuite déduit des étiquettes quand c'est possible.
    """
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    llm = None
    out: dict[str, dict] = {}
    for rec in records:
        lines = outcome_lines(rec)
        if not any(k in lines for k in ("sensitivity", "specificity", "auc", "other_outcomes")):
            out[rec.paper_id] = {"valid": [], "rejected": [], "note": "no verified outcome data"}
            continue
        sig = hashlib.sha256((SYSTEM_SWIM + RETRY + EXAMPLE + json.dumps(lines, sort_keys=True)).encode()).hexdigest()[:12]
        if cache.get(rec.paper_id, {}).get("signature") == sig:
            out[rec.paper_id] = cache[rec.paper_id]
            continue
        llm = llm or llm_factory()
        print(f"  [LLM] comparaisons {rec.paper_id}...", flush=True)
        valid: dict[tuple, Comparison] = {}
        rejected: list[dict] = []
        try:
            first = _ask(rec.paper_id, lines, llm).comparisons
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] SWiM : échec %s", rec.paper_id, e)
            first = []
        problems = []
        for c in first:
            why = validate(c, lines)
            if why:
                problems.append(why)
                rejected.append({**c.model_dump(), "reason": why})
            else:
                valid[_key(c)] = c
        covered = {c.outcome for c in valid.values()}
        for o in hints(lines):
            if o not in covered:
                problems.append(f"the {o} line contains a value for an AI arm and a value for a comparator arm, "
                                "but no comparison was returned for it")
        if problems:
            print(f"  [LLM] {rec.paper_id} : {len(problems)} problème(s), nouvel essai...", flush=True)
            try:
                second = _ask(rec.paper_id, lines, llm, RETRY.format(problems="\n".join(f"- {p}" for p in problems))).comparisons
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] SWiM (2e essai) : échec %s", rec.paper_id, e)
                second = []
            for c in second:
                why = validate(c, lines)
                if why:
                    rejected.append({**c.model_dump(), "reason": why, "attempt": 2})
                else:
                    valid.setdefault(_key(c), c)
        accepted = []
        for c in valid.values():
            role = role_from_labels(c)
            accepted.append({**c.model_dump(), "ai_role": role, "model_role": c.ai_role})
        out[rec.paper_id] = {"signature": sig, "valid": accepted, "rejected": rejected}
        cache[rec.paper_id] = out[rec.paper_id]
        cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- dénombrement SWiM
def sign_test(n_pos: int, n_neg: int) -> float | None:
    """Test du signe exact bilatéral (ex aequo exclus)."""
    n = n_pos + n_neg
    if n == 0:
        return None
    k = min(n_pos, n_neg)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def study_direction(dirs: list[str]) -> str:
    """Direction d'une étude pour une issue : unanime, sinon « mixte » (◐) — SWiM compte des études, pas des comparaisons."""
    s = set(dirs)
    return dirs[0] if len(s) == 1 else "◐"


def tally(comparisons: dict[str, list[Comparison]], threshold: float) -> dict[tuple[str, str], dict[str, str]]:
    """{(rôle, issue): {paper_id: direction de l'étude}}."""
    by: dict[tuple[str, str], dict[str, list[str]]] = {}
    for pid, comps in comparisons.items():
        for c in comps:
            by.setdefault((c.ai_role, c.outcome), {}).setdefault(pid, []).append(direction(c, threshold))
    return {k: {pid: study_direction(d) for pid, d in v.items()} for k, v in by.items()}


# --------------------------------------------------------------------------- rendu (entièrement par le code)
def _ids(ids: list[str]) -> str:
    return "[" + ", ".join(sorted(ids)) + "]"


def _fmt_p(p: float | None) -> str:
    return "not applicable" if p is None else (f"p = {p:.2f}" if p >= 0.01 else "p < 0.01")


def render(results: dict[str, dict], facts: dict[str, dict], threshold: float) -> dict[str, str]:
    comps = {pid: [Comparison.model_validate(c) for c in r["valid"]] for pid, r in results.items()}
    t = tally(comps, threshold)
    n_valid = sum(len(v) for v in comps.values())
    n_rej = sum(len(r.get("rejected", [])) for r in results.values())
    with_data = sorted(pid for pid, v in comps.items() if v)
    without = sorted(pid for pid, v in comps.items() if not v)

    def caveat(pid: str) -> str:
        f = facts.get(pid, {})
        bits = []
        if f.get("level") == "abstract_only":
            bits.append("abstract only")
        if f.get("overall"):
            bits.append(f"{f['overall']} risk of bias")
        return ", ".join(bits)

    # tableau de direction des effets
    cols = [o for o in OUTCOMES if any(k[1] == o for k in t)]
    head = ["Study", "AI role", *[OUTCOMES[o]["label"] for o in cols], "Evidence", "Overall RoB"]
    rows = []
    for pid in with_data:
        for role in ROLES:
            cells = [t.get((role, o), {}).get(pid, "") for o in cols]
            if any(cells):
                f = facts.get(pid, {})
                rows.append([pid, role, *cells, "abstract" if f.get("level") == "abstract_only" else "full text", f.get("overall", "")])
    table = "| " + " | ".join(head) + " |\n|" + "|".join(["---"] * len(head)) + "|\n"
    table += "\n".join("| " + " | ".join(r) + " |" for r in rows)

    # texte par rôle et par issue
    parts = []
    for role, desc in ROLES.items():
        keys = [k for k in t if k[0] == role]
        if not keys:
            continue
        lines = [f"**{desc[0].upper() + desc[1:]}.**"]
        for o in OUTCOMES:
            d = t.get((role, o))
            if not d:
                continue
            pos = [p for p, x in d.items() if x == "▲"]
            neg = [p for p, x in d.items() if x == "▼"]
            nod = [p for p, x in d.items() if x == "◆"]
            mix = [p for p, x in d.items() if x == "◐"]
            seg = []
            if pos:
                seg.append(f"favoured the AI arm in {len(pos)} {_ids(pos)}")
            if nod:
                seg.append(f"showed no clear difference in {len(nod)} {_ids(nod)}")
            if neg:
                seg.append(f"favoured the comparator in {len(neg)} {_ids(neg)}")
            if mix:
                seg.append(f"was mixed across comparisons in {len(mix)} {_ids(mix)}")
            n = len(d)
            lines.append(
                f"- *{OUTCOMES[o]['label']}* ({n} {'study' if n == 1 else 'studies'}): the direction of effect "
                + "; ".join(seg) + f" (sign test {_fmt_p(sign_test(len(pos), len(neg)))})."
            )
            for pid in sorted(d):
                for c in comps[pid]:
                    if c.ai_role == role and c.outcome == o:
                        lines.append(
                            f"  - {pid} ({caveat(pid)}): {c.intervention_label} {c.intervention_value} vs "
                            f"{c.comparator_label} {c.comparator_value} {direction(c, threshold)}"
                        )
        parts.append("\n".join(lines))
    synthesis = "\n\n".join(parts) if parts else "_No head-to-head comparison could be verified._"

    # conclusions (gabarit à partir des décomptes)
    concl = []
    role_name = {"standalone": "AI alone", "assisted": "AI-assisted reading", "triage": "AI triage"}
    for (role, o), d in sorted(t.items()):
        if len(d) < 2:
            continue
        counts = [(sum(x == s for x in d.values()), txt) for s, txt in
                  (("▲", "favoured AI"), ("◆", "showed no clear difference"), ("▼", "favoured the comparator"), ("◐", "were mixed"))]
        parts_ = [f"{n} {txt}" for n, txt in counts if n]
        concl.append(f"{OUTCOMES[o]['label']} with {role_name[role]} ({len(d)} studies): " + ", ".join(parts_))
    hi = sum(1 for p in with_data if facts.get(p, {}).get("overall") == "high")
    ab = sum(1 for p in with_data if facts.get(p, {}).get("level") == "abstract_only")
    conclusion = (
        f"Verified head-to-head comparisons were available for {len(with_data)} of {len(comps)} included studies "
        f"({n_valid} comparisons; {n_rej} candidate comparisons rejected by the automatic checks). "
        + (("Outcomes compared in at least two studies: " + "; ".join(concl) + ". ") if concl else
           "No outcome was compared in more than one study, so no pattern across studies can be described. ")
        + f"Of the studies contributing comparisons, {hi} were at high overall risk of bias and {ab} were assessed on "
        f"the abstract only. With so few studies, vote counting by direction of effect cannot establish whether AI "
        f"performs better or worse than radiologists; sign tests are reported for transparency only. The evidence "
        f"is insufficient to draw a firm conclusion and the findings must be confirmed with a complete search and "
        f"full-text access."
    )
    note = ("▲ favours the AI arm; ▼ favours the comparator; ◆ no clear difference (relative difference below "
            f"{threshold:.0%} of the comparator value); ◐ mixed directions within the study. Directions are computed "
            "by code from the verified values; statistical significance within studies is not considered.")
    return {
        "table": table, "note": note, "synthesis": synthesis, "conclusion": conclusion, "outcome_counts": concl,
        "without": ", ".join(without), "n_valid": n_valid, "n_rejected": n_rej, "studies_with_data": len(with_data),
    }
