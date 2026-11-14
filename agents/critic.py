"""Critic / QA Agent : contrôle qualité de tout le run, sans LLM.

Choix assumé : les essais de ce projet ont montré qu'un LLM de 9B qui relit son propre travail ne détecte pas ses
erreurs de sens. Le critique vérifie donc ce qui est vérifiable mécaniquement, et liste ce qu'un humain doit relire :
- cohérence des effectifs PRISMA d'une étape à l'autre (et avec le rapport et le manuscrit) ;
- respect du protocole (période, codes de critères, décisions « maybe » en attente) ;
- relectures humaines en attente (screening, éligibilité) ;
- accord entre relecteurs simulés ;
- valeurs extraites non étayées, jugements de biais sans citation vérifiée ;
- synthèse SWiM (comparaisons rejetées, études sans comparaison) ;
- livrables périmés (manuscrit plus ancien que des décisions humaines) ;
- erreurs du LLM dans la trace.
Sorties : report/qa_report.md et report/qa.json. Niveaux : error (à corriger), warning (à relire), info.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from agents.observability import run_summary, summary_md
from agents.schemas import Protocol

LEVELS = ("error", "warning", "info")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class QA:
    def __init__(self) -> None:
        self.findings: list[dict] = []

    def add(self, level: str, area: str, message: str) -> None:
        self.findings.append({"level": level, "area": area, "message": message})


def _codes_ok(code: str, protocol: Protocol) -> bool:
    for part in (code or "").split("/"):
        part = part.strip()
        if not part or part in ("none", "human", "unspecified") or part.startswith("RULE:"):
            continue
        m = re.fullmatch(r"([IE])(\d+)", part.upper())
        if not m:
            return False
        n = int(m.group(2))
        limit = len(protocol.inclusion_criteria) if m.group(1) == "I" else len(protocol.exclusion_criteria)
        if not 1 <= n <= limit:
            return False
    return True


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_flow(run_dir: Path, qa: QA) -> dict | None:
    from agents.prisma import collect_flow

    try:
        flow = collect_flow(run_dir)
    except SystemExit as e:
        qa.add("info", "PRISMA", f"Flow incomplete: {e}")
        return None
    n_before = sum(f["level"] == "error" for f in qa.findings)
    for w in flow["warnings"]:
        qa.add("error", "PRISMA", w)
    passed = _load(run_dir / "screening" / "included_for_fulltext.json")
    if passed is not None and len(passed) != flow["screening"]["passed"]:
        qa.add("error", "PRISMA", f"{flow['screening']['passed']} records passed screening but included_for_fulltext.json lists {len(passed)}.")
    bib = run_dir / "report" / "references.bib"
    if bib.exists():
        n = sum(1 for l in bib.read_text(encoding="utf-8").splitlines() if l.startswith("@"))
        if n != flow["included"]["studies"]:
            qa.add("error", "Report", f"references.bib has {n} entries for {flow['included']['studies']} included studies (rerun --report-only).")
    ms = run_dir / "report" / "manuscript.md"
    if ms.exists():
        m = re.search(r"\*\*(\d+) studies were included\*\*", ms.read_text(encoding="utf-8"))
        if m and int(m.group(1)) != flow["included"]["studies"]:
            qa.add("error", "Manuscript", f"Manuscript says {m.group(1)} studies were included, flow says {flow['included']['studies']} (rerun --manuscript-only).")
    if sum(f["level"] == "error" for f in qa.findings) == n_before:
        i = flow["identification"]
        qa.add("info", "PRISMA", f"Counts consistent: {i['total_retrieved']} identified, {i['unique_records']} screened, "
               f"{flow['eligibility']['assessed']} assessed, {flow['included']['studies']} included.")
    return flow


def check_protocol(run_dir: Path, protocol: Protocol, qa: QA) -> None:
    included = _load(run_dir / "fulltext" / "included_for_extraction.json") or []
    lo = int(protocol.date_from) if protocol.date_from.isdigit() else None
    hi = int(protocol.date_to) if protocol.date_to.isdigit() else None
    out = [d["id"] for d in included if d.get("year") and ((lo and d["year"] < lo) or (hi and d["year"] > hi))]
    if out:
        qa.add("error", "Protocol", f"Included studies outside the protocol period: {', '.join(out)}.")
    maybe = [d["id"] for d in included if d.get("eligibility") == "maybe"]
    if maybe:
        qa.add("warning", "Protocol", f"{len(maybe)} included studies still 'maybe' (human decision pending): {', '.join(maybe)}.")
    for stage, path in (("screening", run_dir / "screening" / "decisions.jsonl"), ("eligibility", run_dir / "fulltext" / "eligibility.jsonl")):
        bad = sorted({r["final_criterion"] for r in _jsonl(path) if r.get("final_decision") == "exclude" and not _codes_ok(r.get("final_criterion", ""), protocol)})
        if bad:
            qa.add("warning", "Protocol", f"{stage}: exclusion codes not in the protocol: {', '.join(bad)}.")


def check_reviews(run_dir: Path, qa: QA) -> None:
    from agents.review import review_queue

    for stage, folder in (("screening", "screening"), ("eligibility", "fulltext")):
        try:
            q = review_queue(run_dir, stage)
        except SystemExit:
            continue
        if q:
            ids = ", ".join(r.paper_id for r in q[:12]) + (" ..." if len(q) > 12 else "")
            qa.add("warning", "Human review", f"{stage}: {len(q)} decisions flagged for human review without a human decision "
                   f"({ids}). Run: python main.py --resume <RUN> --review {stage}")
    s = _load(run_dir / "screening" / "prisma_screening.json") or {}
    e = _load(run_dir / "fulltext" / "prisma_eligibility.json") or {}
    for name, d in (("screening", s), ("eligibility", e)):
        k = d.get("cohens_kappa")
        if k is not None:
            qa.add("warning" if k < 0.6 else "info", "Agreement",
                   f"{name}: Cohen's kappa between simulated reviewers = {k} (same model: consistency, not accuracy).")
        if d.get("errors"):
            qa.add("error", "Pipeline", f"{name}: {d['errors']} automatic decisions failed (decided_by = error).")


def check_evidence(run_dir: Path, qa: QA) -> None:
    from agents.evidence import current_records

    try:
        records, problems = current_records(run_dir)
    except SystemExit:
        return
    for p in problems:
        qa.add("error", "Extraction", p)
    for r in records:
        if r.error:
            qa.add("error", "Extraction", f"{r.paper_id}: {r.error}")
        unv = [n for n, fv in r.extracted.items() if fv.found and not fv.grounded]
        if unv:
            qa.add("warning", "Extraction", f"{r.paper_id}: unverified values ({', '.join(unv)}) — check against the text.")
        for d in r.risk_of_bias:
            if d.judgement in ("low", "high") and not d.evidence_verified and not d.human:
                qa.add("error", "Risk of bias", f"{r.paper_id} {d.domain}: '{d.judgement}' without a verified quote.")
        if r.evidence_level == "full_text" and r.risk_of_bias and all(d.judgement == "unclear" for d in r.risk_of_bias):
            qa.add("warning", "Risk of bias", f"{r.paper_id}: full text available but every domain is 'unclear' — review manually.")
    found = [fv for r in records for fv in r.extracted.values() if fv.found]
    if found:
        rate = sum(fv.grounded for fv in found) / len(found)
        qa.add("warning" if rate < 0.9 else "info", "Extraction", f"{rate:.1%} of extracted values are verified or human-corrected.")


def check_synthesis(run_dir: Path, qa: QA) -> None:
    sw = _load(run_dir / "report" / "swim.json")
    if sw is None:
        qa.add("info", "Synthesis", "No SWiM synthesis yet (run --manuscript-only).")
        return
    valid = sum(len(v.get("valid", [])) for v in sw.values())
    rej = sum(len(v.get("rejected", [])) for v in sw.values())
    none = sorted(k for k, v in sw.items() if not v.get("valid"))
    qa.add("info", "Synthesis", f"SWiM: {valid} verified comparisons, {rej} rejected by checks.")
    if none:
        qa.add("warning", "Synthesis", f"No verified comparison for {', '.join(none)}: check whether the data contain one "
               "(e.g. a value written without a group label) and correct via evidence/human_overrides.csv.")


def check_staleness(run_dir: Path, qa: QA) -> None:
    inputs = [run_dir / p for p in ("screening/human_overrides.csv", "fulltext/human_overrides.csv",
                                    "fulltext/included_for_extraction.json", "evidence/records.jsonl",
                                    "evidence/human_overrides.csv")]
    latest = max((p.stat().st_mtime for p in inputs if p.exists()), default=0)
    for name, cmd in (("report/tables.md", "--report-only"), ("report/manuscript.md", "--manuscript-only")):
        p = run_dir / name
        if p.exists() and p.stat().st_mtime < latest:
            qa.add("warning", "Deliverables", f"{name} is older than the latest decisions or extraction (rerun {cmd}).")


def check_trace(summary: dict, qa: QA) -> None:
    t = summary["llm"]
    echo = sum(v["schema_echo"] for v in t["agents"].values())
    qa.add("info", "LLM", f"{t['total_calls']} successful LLM calls, {t['total_errors']} failed attempts "
           f"(of which {echo} returned the JSON schema), {t['total_seconds']} s of LLM time.")


def run_qa(run_dir: Path, protocol: Protocol) -> dict:
    qa = QA()
    check_flow(run_dir, qa)
    check_protocol(run_dir, protocol, qa)
    check_reviews(run_dir, qa)
    check_evidence(run_dir, qa)
    check_synthesis(run_dir, qa)
    check_staleness(run_dir, qa)
    obs = run_summary(run_dir)
    check_trace(obs, qa)

    counts = {lv: sum(f["level"] == lv for f in qa.findings) for lv in LEVELS}
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    result = {"generated": datetime.now().isoformat(timespec="seconds"), "counts": counts,
              "findings": qa.findings, "observability": obs}
    (out / "qa.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    icon = {"error": "❌", "warning": "⚠", "info": "ℹ"}
    lines = [f"# Quality assurance report", "",
             f"Generated {result['generated']} — {counts['error']} errors, {counts['warning']} warnings, {counts['info']} notes.", ""]
    for lv in LEVELS:
        items = [f for f in qa.findings if f["level"] == lv]
        if items:
            lines += [f"## {lv.capitalize()}s", ""] + [f"- {icon[lv]} **{f['area']}** — {f['message']}" for f in items] + [""]
    lines += ["## Observability (LLM calls from trace.jsonl)", "", summary_md(obs), ""]
    (out / "qa_report.md").write_text("\n".join(lines), encoding="utf-8")
    result["path"] = str(out / "qa_report.md")
    return result


def print_qa(res: dict) -> None:
    c = res["counts"]
    print("\n=== CONTRÔLE QUALITÉ (agent critique) ===")
    print(f"  Erreurs : {c['error']} | À relire : {c['warning']} | Notes : {c['info']}")
    for f in res["findings"]:
        if f["level"] != "info":
            print(f"  [{f['level']}] {f['area']} : {f['message']}")
    print(f"  Rapport : {res['path']}")
