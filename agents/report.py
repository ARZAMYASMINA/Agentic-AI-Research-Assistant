"""Livrables déterministes du rapport (sans LLM) : diagramme PRISMA, références BibTeX, tableaux Markdown.

Écrit dans <run>/report/ : prisma_flow.png, prisma_flow.json, references.bib, tables.md.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from agents.prisma import collect_flow, render_png
from agents.schemas import EvidenceRecord
from tools.bibtex import to_bibtex
from tools.paper import Paper


def _cell(value, limit: int = 140) -> str:
    s = " ".join(str(value).split()).replace("|", "/")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def load_evidence(run_dir: Path) -> list[EvidenceRecord]:
    """Études actuellement incluses, avec les corrections humaines de evidence/human_overrides.csv."""
    from agents.evidence import current_records  # import tardif : évite un cycle d'imports

    return current_records(run_dir)[0]


def _value(rec: EvidenceRecord, name: str, limit: int = 140) -> str:
    fv = rec.extracted.get(name)
    if fv is None or not fv.found:
        return "not reported"
    flag = " †" if fv.human else ("" if fv.grounded else " ⚠")  # † corrigé par un humain ; ⚠ non étayé : à contrôler
    return _cell(fv.value, limit) + flag


def characteristics_md(records: list[EvidenceRecord]) -> str:
    rows = [
        [r.paper_id, str(r.year or ""), _value(r, "study_design", 60), _value(r, "population", 120),
         _value(r, "ai_system", 100), _value(r, "comparator", 100), r.evidence_level.replace("_", " ")]
        for r in records
    ]
    return _table(["ID", "Year", "Design", "Population", "AI system", "Comparator", "Evidence"], rows)


def results_md(records: list[EvidenceRecord]) -> str:
    rows = [
        [r.paper_id, _value(r, "sensitivity"), _value(r, "specificity"), _value(r, "auc"), _value(r, "other_outcomes", 160)]
        for r in records
    ]
    return _table(["ID", "Sensitivity", "Specificity", "AUC", "Other outcomes"], rows)


def rob_md(records: list[EvidenceRecord]) -> str:
    domains = [d.domain for d in records[0].risk_of_bias] if records and records[0].risk_of_bias else []
    rows = []
    for r in records:
        by = {d.domain: d.judgement + (" †" if d.human else "") for d in r.risk_of_bias}
        rows.append([r.paper_id, *[by.get(d, "unclear") for d in domains], r.overall_risk])
    return _table(["ID", *[d.replace("_", " ") for d in domains], "Overall"], rows)


def excluded_md(run_dir: Path) -> str:
    path = run_dir / "fulltext" / "eligibility_results.csv"
    if not path.exists():
        return "_No eligibility results available._"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    excluded = [r for r in rows if r["final_decision"] == "exclude" and r["criterion"] != "RULE:not_retrieved"]
    not_retrieved = [r for r in rows if r["criterion"] == "RULE:not_retrieved"]
    parts = ["**Reports excluded after full-text assessment**\n", _table(
        ["ID", "Title", "Reason for exclusion"], [[r["id"], _cell(r["title"], 110), _cell(r["reason"], 110)] for r in excluded]
    )]
    if not_retrieved:
        parts += ["\n**Reports not retrieved** (no open-access full text or abstract)\n",
                  _table(["ID", "Title"], [[r["id"], _cell(r["title"], 130)] for r in not_retrieved])]
    return "\n".join(parts)


def write_report_data(run_dir: Path) -> dict:
    out = run_dir / "report"
    out.mkdir(exist_ok=True)

    flow = collect_flow(run_dir)
    (out / "prisma_flow.json").write_text(json.dumps(flow, indent=2, ensure_ascii=False), encoding="utf-8")
    render_png(flow, out / "prisma_flow.png")

    records = load_evidence(run_dir)
    if len(records) != flow["included"]["studies"]:
        flow["warnings"].append(
            f"{flow['included']['studies']} études incluses mais {len(records)} extraites (relancez --extract-only)."
        )
    metas = {d["id"]: d for d in json.loads((run_dir / "fulltext" / "included_for_extraction.json").read_text(encoding="utf-8"))}
    papers = [Paper.model_validate(metas[r.paper_id]) for r in records if r.paper_id in metas]
    (out / "references.bib").write_text(to_bibtex(papers), encoding="utf-8")

    note = ("⚠ = valeur non étayée par le texte (ni citation exacte, ni chiffres retrouvés) : à contrôler à la main. "
            "† = valeur corrigée par un humain (evidence/human_overrides.csv).")
    tables = "\n\n".join([
        "## Study characteristics\n\n" + characteristics_md(records),
        "## Results of individual studies\n\n" + results_md(records) + f"\n\n_{note}_",
        "## Risk of bias (automatic draft, to be verified by a human)\n\n" + rob_md(records)
        + "\n\n_† = judgement corrected by a human reviewer (evidence/human_overrides.csv)._",
        "## Excluded and unretrieved reports\n\n" + excluded_md(run_dir),
    ])
    (out / "tables.md").write_text(tables + "\n", encoding="utf-8")
    return {"flow": flow, "studies": len(records), "references": len(papers), "dir": str(out)}


def print_report(res: dict) -> None:
    f = res["flow"]
    print("\n=== LIVRABLES DU RAPPORT ===")
    print(f"  Diagramme PRISMA   : {res['dir']}\\prisma_flow.png")
    print(f"  Références BibTeX  : {res['dir']}\\references.bib ({res['references']} études)")
    print(f"  Tableaux           : {res['dir']}\\tables.md")
    print(f"  Études incluses    : {f['included']['studies']} (texte intégral {f['included']['full_text']}, résumé seul {f['included']['abstract_only']})")
    for w in f["warnings"]:
        print(f"  ATTENTION : {w}")
