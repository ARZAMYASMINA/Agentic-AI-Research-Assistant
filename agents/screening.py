"""Screening Agents (titre/résumé) : double revue simulée + arbitre, avec justification tracée.

- Relecteur A (favorise le rappel) et relecteur B (favorise la précision, critères listés dans l'ordre inverse)
  décident indépendamment, avec les mêmes critères numérotés I1..In / E1..En.
- Accord (ou include/maybe) : décision de consensus. Désaccord include|maybe vs exclude : l'arbitre tranche.
- Pré-filtre déterministe (sans LLM) : année hors période du protocole.
- Chaque décision est ajoutée à screening/decisions.jsonl : un run interrompu reprend où il s'est arrêté.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from agents.human import apply_override, load_overrides
from agents.llm import get_llm, structured_call
from agents.schemas import Protocol, ScreeningDecision, ScreeningRecord
from tools.metrics import cohen_kappa
from tools.paper import Paper

log = logging.getLogger(__name__)

MAX_ABSTRACT_CHARS = 2500

SYSTEM_REVIEWER = """You are {who}, screening titles and abstracts for a systematic review.
Decide from the title and abstract ONLY, applying the eligibility criteria given by the user.
Rules:
- "include": the record clearly meets the inclusion criteria and no exclusion criterion applies.
- "exclude": at least one exclusion criterion clearly applies, or an essential inclusion criterion clearly fails.
- "maybe": the information is insufficient to decide (for example, no abstract). Never guess:
  "maybe" sends the record to full-text review.
- {stance}
- `criterion`: the code of the main criterion behind your decision (for example "I2" or "E3"),
  or "none" for a clean include.
- `rationale`: 1-2 sentences citing specific details from the title/abstract. Write it BEFORE the decision.
- The language of the text may be used to apply language criteria."""

STANCE_A = "When in doubt between exclude and maybe, choose maybe (favor recall)."
STANCE_B = "Check every criterion one by one and exclude when a criterion is clearly not met (favor precision)."

SYSTEM_ARBITER = """You are a senior methodologist resolving a disagreement between two reviewers who screen
titles and abstracts for a systematic review. Read the criteria, the record and both reviewers' decisions,
then give the final decision: include, exclude, or maybe (maybe = insufficient information, goes to full-text review).
Be conservative at this stage: exclude ONLY when the title/abstract explicitly shows that a criterion fails.
If your reason would rest on speculation (for example "may", "suggests", "potentially"), choose maybe instead.
Judge the evidence in the title/abstract itself, not the reviewers' confidence.
`criterion` is the code of the main criterion (for example "I2" or "E3"), or "none" for a clean include.
Write the rationale (1-2 sentences, citing details of the record) BEFORE the decision."""


# --------------------------------------------------------------------------- prompts
def _header(p: Protocol) -> str:
    return (
        f"Review question: {p.research_question}\n"
        f"Population: {p.pico.population}\nIntervention: {p.pico.intervention}\n"
        f"Comparator: {p.pico.comparator or 'not specified'}\nOutcome: {p.pico.outcome}"
    )


def _criteria_text(p: Protocol, reverse: bool = False) -> str:
    inc = [f"I{i}. {c}" for i, c in enumerate(p.inclusion_criteria, 1)]
    exc = [f"E{i}. {c}" for i, c in enumerate(p.exclusion_criteria, 1)]
    if reverse:  # relecteur B : ordre inverse, mêmes codes
        inc, exc = inc[::-1], exc[::-1]
    return "Inclusion criteria:\n" + "\n".join(inc) + "\n\nExclusion criteria:\n" + "\n".join(exc)


def _paper_text(paper: Paper) -> str:
    abstract = paper.abstract[:MAX_ABSTRACT_CHARS] or "(no abstract available)"
    return (
        f"ID: {paper.id}\nYear: {paper.year or 'unknown'}\nJournal: {paper.journal or 'unknown'}\n"
        f"Title: {paper.title}\nAbstract: {abstract}"
    )


# --------------------------------------------------------------------------- agents
def _review(paper: Paper, protocol: Protocol, llm, who: str) -> ScreeningDecision:
    b = who == "B"
    system = SYSTEM_REVIEWER.format(who=f"independent reviewer {who}", stance=STANCE_B if b else STANCE_A)
    user = (
        f"{_header(protocol)}\n\n{_criteria_text(protocol, reverse=b)}\n\n"
        f"Record to screen:\n{_paper_text(paper)}\n\nGive your screening decision."
    )
    return structured_call(ScreeningDecision, system, user, agent=f"screening_{who.lower()}", llm=llm)


def _arbitrate(paper: Paper, protocol: Protocol, llm, a: ScreeningDecision, b: ScreeningDecision) -> ScreeningDecision:
    def fmt(name: str, d: ScreeningDecision) -> str:
        return f"Reviewer {name}: decision={d.decision} (criterion {d.criterion}, confidence {d.confidence}). Rationale: {d.rationale}"

    user = (
        f"{_header(protocol)}\n\n{_criteria_text(protocol)}\n\nRecord:\n{_paper_text(paper)}\n\n"
        f"{fmt('A', a)}\n{fmt('B', b)}\n\nGive the final decision."
    )
    return structured_call(ScreeningDecision, SYSTEM_ARBITER, user, agent="screening_arbiter", llm=llm)


def rule_check(paper: Paper, protocol: Protocol) -> ScreeningRecord | None:
    """Exclusion déterministe (sans LLM) : année hors période du protocole."""
    try:
        lo = int(protocol.date_from) if protocol.date_from else None
        hi = int(protocol.date_to) if protocol.date_to else None
    except ValueError:
        return None
    if paper.year and ((lo and paper.year < lo) or (hi and paper.year > hi)):
        return ScreeningRecord(
            paper_id=paper.id,
            title=paper.title,
            final_decision="exclude",
            final_criterion="RULE:date",
            final_rationale=f"Published in {paper.year}, outside the protocol period ({lo or '...'}-{hi or '...'}).",
            decided_by="rule",
        )
    return None


def combine(a: ScreeningDecision, b: ScreeningDecision) -> str | None:
    """Décision de consensus, ou None si l'arbitre doit trancher (exclude contre include/maybe)."""
    if a.decision == b.decision:
        return a.decision
    if "exclude" in (a.decision, b.decision):
        return None
    return "maybe"  # include contre maybe : on laisse passer au texte intégral


def screen_one(paper: Paper, protocol: Protocol, llm) -> ScreeningRecord:
    ruled = rule_check(paper, protocol)
    if ruled:
        return ruled

    a = _review(paper, protocol, llm, "A")
    b = _review(paper, protocol, llm, "B")
    merged = combine(a, b)

    if merged is not None:
        crit = a.criterion if a.criterion == b.criterion else f"{a.criterion}/{b.criterion}"
        return ScreeningRecord(
            paper_id=paper.id,
            title=paper.title,
            reviewer_a=a,
            reviewer_b=b,
            final_decision=merged,
            final_criterion=crit,
            final_rationale=f"A: {a.rationale} | B: {b.rationale}",
            decided_by="consensus",
            needs_human_review=merged == "maybe" or (a.confidence == "low" and b.confidence == "low"),
        )

    arb = _arbitrate(paper, protocol, llm, a, b)
    return ScreeningRecord(
        paper_id=paper.id,
        title=paper.title,
        reviewer_a=a,
        reviewer_b=b,
        arbiter=arb,
        final_decision=arb.decision,
        final_criterion=arb.criterion,
        final_rationale=arb.rationale,
        decided_by="arbiter",
        needs_human_review=True,  # un désaccord entre relecteurs est toujours à faire vérifier par un humain
    )


# --------------------------------------------------------------------------- pipeline
def _load_papers(run_dir: Path) -> list[Paper]:
    data = json.loads((run_dir / "records_deduplicated.json").read_text(encoding="utf-8"))
    return [Paper.model_validate(d) for d in data]


def summarize(records: list[ScreeningRecord]) -> dict:
    final = Counter(r.final_decision for r in records)
    dual = [r for r in records if r.reviewer_a and r.reviewer_b]
    a_pass = [r.reviewer_a.decision != "exclude" for r in dual]
    b_pass = [r.reviewer_b.decision != "exclude" for r in dual]
    kappa = cohen_kappa(a_pass, b_pass)
    agree = sum(x == y for x, y in zip(a_pass, b_pass)) / len(dual) if dual else None
    return {
        "records_screened": len(records),
        "excluded_by_rule": sum(r.decided_by == "rule" for r in records),
        "included": final["include"],
        "maybe": final["maybe"],
        "excluded": final["exclude"],
        "passed_to_fulltext": final["include"] + final["maybe"],
        "arbiter_calls": sum(r.decided_by == "arbiter" for r in records),
        "needs_human_review": sum(r.needs_human_review for r in records),
        "human_decisions": sum(r.decided_by == "human" for r in records),
        "errors": sum(r.decided_by == "error" for r in records),
        "dual_screened": len(dual),
        "inter_reviewer_agreement": round(agree, 3) if agree is not None else None,
        "cohens_kappa": round(kappa, 3) if kappa is not None else None,
        "exclusion_reasons": dict(Counter(r.final_criterion or "unspecified" for r in records if r.final_decision == "exclude")),
    }


def _write_csv(path: Path, records: list[ScreeningRecord], years: dict[str, int | None]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig : s'ouvre correctement dans Excel
        w = csv.writer(f)
        w.writerow(["id", "year", "title", "final_decision", "decided_by", "criterion", "needs_human_review",
                    "reviewer_A", "reviewer_B", "arbiter", "rationale"])
        for r in records:
            w.writerow([
                r.paper_id, years.get(r.paper_id) or "", r.title, r.final_decision, r.decided_by, r.final_criterion,
                "yes" if r.needs_human_review else "",
                r.reviewer_a.decision if r.reviewer_a else "", r.reviewer_b.decision if r.reviewer_b else "",
                r.arbiter.decision if r.arbiter else "", r.final_rationale,
            ])


def run_screening(run_dir: Path, protocol: Protocol, limit: int | None = None) -> dict:
    papers = _load_papers(run_dir)
    todo = papers[:limit] if limit else papers
    out_dir = run_dir / "screening"
    out_dir.mkdir(exist_ok=True)
    jsonl = out_dir / "decisions.jsonl"

    # Si le protocole ou les prompts ont changé, les anciennes décisions ne sont plus valables : on les archive
    material = json.dumps(protocol.model_dump(), sort_keys=True) + SYSTEM_REVIEWER + STANCE_A + STANCE_B + SYSTEM_ARBITER
    signature = hashlib.sha256(material.encode()).hexdigest()[:12]
    sig_file = out_dir / "protocol_signature.txt"
    if jsonl.exists() and (not sig_file.exists() or sig_file.read_text().strip() != signature):
        backup = out_dir / f"decisions.{datetime.now():%Y%m%d_%H%M%S}.bak.jsonl"
        jsonl.rename(backup)
        print(f"  Protocole ou prompts modifiés (ou signature absente) : anciennes décisions archivées dans {backup.name}")
    sig_file.write_text(signature)

    done: dict[str, ScreeningRecord] = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = ScreeningRecord.model_validate_json(line)
                done[rec.paper_id] = rec
        if done:
            print(f"  Reprise : {len(done)} articles déjà screenés.")

    llm = get_llm()
    t0, new = time.time(), 0
    for i, paper in enumerate(todo, 1):
        if paper.id in done:
            continue
        try:
            rec = screen_one(paper, protocol, llm)
        except Exception as e:  # noqa: BLE001  un échec isolé ne doit pas arrêter tout le screening
            log.error("[%s] screening échoué : %s", paper.id, e)
            rec = ScreeningRecord(
                paper_id=paper.id, title=paper.title, final_decision="maybe",
                final_rationale=f"Automatic screening failed ({str(e)[:200]}); needs human review.",
                decided_by="error", needs_human_review=True,
            )
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")
        done[paper.id] = rec
        new += 1
        remaining = sum(1 for p in todo if p.id not in done)
        eta = (time.time() - t0) / new * remaining
        print(f"  [{i}/{len(todo)}] {paper.id} {rec.final_decision:<7} ({rec.decided_by}, {rec.final_criterion or '-'}) "
              f"ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)

    overrides = load_overrides(out_dir)
    unknown = sorted(set(overrides) - {p.id for p in todo})
    if unknown:
        print(f"  Attention : identifiants inconnus dans screening/human_overrides.csv (ignorés) : {', '.join(unknown)}")
    records = [apply_override(done[p.id], *overrides[p.id]) if p.id in overrides else done[p.id]
               for p in todo if p.id in done]
    summary = summarize(records)
    (out_dir / "prisma_screening.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out_dir / "results.csv", records, {p.id: p.year for p in papers})

    passed = {r.paper_id for r in records if r.final_decision in ("include", "maybe")}
    (out_dir / "included_for_fulltext.json").write_text(
        json.dumps([p.model_dump() for p in todo if p.id in passed], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def print_screening(s: dict) -> None:
    print("\n=== PRISMA : SCREENING TITRE/RÉSUMÉ ===")
    print(f"  Articles screenés      : {s['records_screened']} (dont {s['excluded_by_rule']} exclus par règle de date)")
    print(f"  Exclus                 : {s['excluded']}")
    print(f"  Passent au texte intégral : {s['passed_to_fulltext']} (include {s['included']}, maybe {s['maybe']})")
    print(f"  Arbitrages             : {s['arbiter_calls']} | À revoir par un humain : {s['needs_human_review']} | Erreurs : {s['errors']}")
    print(f"  Accord A/B             : {s['inter_reviewer_agreement']} | Kappa de Cohen : {s['cohens_kappa']}")
    if s.get("human_decisions"):
        print(f"  Décisions humaines     : {s['human_decisions']} (screening/human_overrides.csv)")
    if s["exclusion_reasons"]:
        print("  Motifs d'exclusion     : " + ", ".join(f"{k}: {v}" for k, v in sorted(s["exclusion_reasons"].items())))
