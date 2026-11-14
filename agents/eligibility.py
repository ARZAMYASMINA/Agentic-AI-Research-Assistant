"""Éligibilité sur texte intégral (PRISMA) : double relecture + arbitre, appuyée par le RAG sur le texte.

Trois niveaux de preuve, explicites dans chaque décision (evidence_level) :
- full_text     : passages pertinents retrouvés dans le texte intégral (index hybride du document)
- abstract_only : seul le résumé est disponible, la décision est donc moins fiable
- none          : ni texte ni résumé -> « rapport non récupéré » (PRISMA), à revoir manuellement
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

from agents.llm import get_llm, structured_call
from agents.schemas import Protocol, ScreeningDecision, ScreeningRecord
from agents.human import apply_override as _apply_override
from agents.human import load_overrides as _load_overrides
from agents.screening import STANCE_A, STANCE_B, _criteria_text, _header, combine
from tools.metrics import cohen_kappa
from tools.paper import Paper

log = logging.getLogger(__name__)

SYSTEM_REVIEWER = """You are {who}, assessing FULL-TEXT eligibility for a systematic review.
You receive the eligibility criteria, the abstract, and EXCERPTS retrieved from the full text.
The excerpts are passages selected by a search engine: a detail missing from them is NOT proof that the
study lacks it. Base your decision on what the text actually says.
First check the study type: is it a primary study (original data) or secondary material (systematic review,
meta-analysis, narrative review, commentary, guideline)? Apply the criteria accordingly.
Rules:
- "include": the study clearly meets all inclusion criteria and no exclusion criterion applies.
- "exclude": an exclusion criterion clearly applies, or an essential inclusion criterion is clearly not met,
  as shown by the text.
- "maybe": the evidence is insufficient or contradictory, a human must decide. If only the abstract is
  available, say so in the rationale and use "maybe" rather than guessing.
- {stance}
- `criterion`: the code of the main criterion behind your decision (for example "I2" or "E3"),
  or "none" for a clean include.
- `rationale`: 1-3 sentences citing specific details of the text. Write it BEFORE the decision."""

SYSTEM_ARBITER = """You are a senior methodologist resolving a disagreement between two reviewers who assess
full-text eligibility for a systematic review. Read the criteria, the evidence and both reviewers' decisions,
then give the final decision: include, exclude, or maybe (maybe = a human must decide).
Exclude ONLY when the text explicitly shows that a criterion fails; if your reason rests on speculation
("may", "suggests", "potentially"), choose maybe. Check the study type (primary vs secondary) first.
`criterion` is the code of the main criterion (for example "I2" or "E3"), or "none" for a clean include.
Write the rationale (1-3 sentences, citing details of the text) BEFORE the decision."""


# --------------------------------------------------------------------------- prompts
def _queries(protocol: Protocol) -> list[str]:
    """Requêtes de recherche dans le texte : question, PICO et chaque critère."""
    base = [
        protocol.research_question,
        "study design, methods, participants and setting",
        protocol.pico.population,
        protocol.pico.intervention,
        protocol.pico.outcome,
    ]
    return [q for q in base + protocol.inclusion_criteria + protocol.exclusion_criteria if q]


def _evidence_block(paper: Paper, level: str, context: str) -> str:
    head = (
        f"ID: {paper.id}\nYear: {paper.year or 'unknown'}\nJournal: {paper.journal or 'unknown'}\n"
        f"Title: {paper.title}\nEvidence level: {level}\n"
        f"Abstract: {paper.abstract[:2500] or '(no abstract available)'}"
    )
    if level == "full_text":
        return f"{head}\n\nExcerpts retrieved from the full text:\n{context}"
    return f"{head}\n\n(Full text not available: only the abstract can be used.)"


def _review(paper: Paper, protocol: Protocol, llm, who: str, level: str, context: str) -> ScreeningDecision:
    b = who == "B"
    system = SYSTEM_REVIEWER.format(who=f"independent reviewer {who}", stance=STANCE_B if b else STANCE_A)
    user = (
        f"{_header(protocol)}\n\n{_criteria_text(protocol, reverse=b)}\n\n"
        f"Record to assess:\n{_evidence_block(paper, level, context)}\n\nGive your eligibility decision."
    )
    return structured_call(ScreeningDecision, system, user, agent=f"eligibility_{who.lower()}", llm=llm)


def _arbitrate(paper, protocol, llm, level, context, a: ScreeningDecision, b: ScreeningDecision) -> ScreeningDecision:
    def fmt(name: str, d: ScreeningDecision) -> str:
        return f"Reviewer {name}: decision={d.decision} (criterion {d.criterion}, confidence {d.confidence}). Rationale: {d.rationale}"

    user = (
        f"{_header(protocol)}\n\n{_criteria_text(protocol)}\n\nRecord:\n{_evidence_block(paper, level, context)}\n\n"
        f"{fmt('A', a)}\n{fmt('B', b)}\n\nGive the final decision."
    )
    return structured_call(ScreeningDecision, SYSTEM_ARBITER, user, agent="eligibility_arbiter", llm=llm)


# --------------------------------------------------------------------------- décision par article
def not_retrieved(paper: Paper) -> ScreeningRecord:
    return ScreeningRecord(
        paper_id=paper.id,
        title=paper.title,
        stage="full_text",
        evidence_level="none",
        final_decision="exclude",
        final_criterion="RULE:not_retrieved",
        final_rationale=(
            "Neither the full text nor an abstract could be retrieved (no open-access version found). "
            f"To assess it, add the PDF as fulltext/pdf/{paper.id}.pdf and rerun."
        ),
        decided_by="rule",
        needs_human_review=True,
    )


def assess_one(paper: Paper, protocol: Protocol, llm, level: str, context: str) -> ScreeningRecord:
    a = _review(paper, protocol, llm, "A", level, context)
    b = _review(paper, protocol, llm, "B", level, context)
    merged = combine(a, b)

    if merged is not None:
        crit = a.criterion if a.criterion == b.criterion else f"{a.criterion}/{b.criterion}"
        return ScreeningRecord(
            paper_id=paper.id, title=paper.title, stage="full_text", evidence_level=level,
            reviewer_a=a, reviewer_b=b, final_decision=merged, final_criterion=crit,
            final_rationale=f"A: {a.rationale} | B: {b.rationale}", decided_by="consensus",
            needs_human_review=merged in ("maybe", "exclude") or level != "full_text" or (a.confidence == "low" and b.confidence == "low"),
        )

    arb = _arbitrate(paper, protocol, llm, level, context, a, b)
    return ScreeningRecord(
        paper_id=paper.id, title=paper.title, stage="full_text", evidence_level=level,
        reviewer_a=a, reviewer_b=b, arbiter=arb, final_decision=arb.decision,
        final_criterion=arb.criterion, final_rationale=arb.rationale, decided_by="arbiter",
        needs_human_review=True,  # un désaccord entre relecteurs est toujours à faire vérifier
    )


# --------------------------------------------------------------------------- rapports
def _flag_exclusions(rec: ScreeningRecord) -> ScreeningRecord:
    """Toute exclusion sur texte intégral est à faire relire par un humain, même en cas de consensus."""
    if rec.final_decision == "exclude" and not rec.needs_human_review:
        return rec.model_copy(update={"needs_human_review": True})
    return rec


def reason_label(code: str, protocol: Protocol) -> str:
    """Traduit un code (I2, E3, RULE:...) en texte lisible pour le diagramme PRISMA."""
    main = code.split("/")[0].strip()
    if main == "RULE:not_retrieved":
        return "Report not retrieved (no open-access full text or abstract)"
    if main == "human":
        return "Excluded on human review"
    kind, num = main[:1].upper(), main[1:]
    if num.isdigit():
        idx = int(num) - 1
        if kind == "I" and 0 <= idx < len(protocol.inclusion_criteria):
            return f"{kind}{num} not met: {_short(protocol.inclusion_criteria[idx])}"
        if kind == "E" and 0 <= idx < len(protocol.exclusion_criteria):
            return f"{kind}{num}: {_short(protocol.exclusion_criteria[idx])}"
    return main or "unspecified"


def _short(text: str, limit: int = 60) -> str:
    """Raccourcit un critère à la frontière d'un mot (« … »), pour un diagramme lisible ; le code renvoie au protocole."""
    text = " ".join(text.split()).rstrip(".")
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-(")
    return f"{cut}…"


def summarize(records: list[ScreeningRecord], protocol: Protocol) -> dict:
    not_ret = [r for r in records if r.final_criterion == "RULE:not_retrieved"]
    assessed = [r for r in records if r.final_criterion != "RULE:not_retrieved"]
    excluded = [r for r in assessed if r.final_decision == "exclude"]
    dual = [r for r in assessed if r.reviewer_a and r.reviewer_b]
    kappa = cohen_kappa(
        [r.reviewer_a.decision != "exclude" for r in dual], [r.reviewer_b.decision != "exclude" for r in dual]
    )
    return {
        "reports_sought": len(records),
        "reports_not_retrieved": len(not_ret),
        "reports_assessed": len(assessed),
        "assessed_on_full_text": sum(r.evidence_level == "full_text" for r in assessed),
        "assessed_on_abstract_only": sum(r.evidence_level == "abstract_only" for r in assessed),
        "excluded": len(excluded),
        "included": sum(r.final_decision == "include" for r in assessed),
        "maybe_human_decision": sum(r.final_decision == "maybe" for r in assessed),
        "needs_human_review": sum(r.needs_human_review for r in records),
        "errors": sum(r.decided_by == "error" for r in records),
        "human_decisions": sum(r.decided_by == "human" for r in records),
        "arbiter_calls": sum(r.decided_by == "arbiter" for r in assessed),
        "cohens_kappa": round(kappa, 3) if kappa is not None else None,
        "exclusion_reasons": dict(Counter(reason_label(r.final_criterion, protocol) for r in excluded)),
    }


def _write_csv(path: Path, records: list[ScreeningRecord], protocol: Protocol) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "evidence_level", "final_decision", "decided_by", "criterion", "reason",
                    "needs_human_review", "reviewer_A", "reviewer_B", "arbiter", "rationale"])
        for r in records:
            w.writerow([
                r.paper_id, r.title, r.evidence_level, r.final_decision, r.decided_by, r.final_criterion,
                reason_label(r.final_criterion, protocol) if r.final_decision == "exclude" else "",
                "yes" if r.needs_human_review else "",
                r.reviewer_a.decision if r.reviewer_a else "", r.reviewer_b.decision if r.reviewer_b else "",
                r.arbiter.decision if r.arbiter else "", r.final_rationale,
            ])


# --------------------------------------------------------------------------- pipeline
def run_eligibility(run_dir: Path, protocol: Protocol) -> dict:
    src = run_dir / "screening" / "included_for_fulltext.json"
    manifest_path = run_dir / "fulltext" / "manifest.json"
    if not src.exists() or not manifest_path.exists():
        raise SystemExit("Lancez d'abord le screening puis la récupération des textes intégraux (--fulltext-only).")
    papers = [Paper.model_validate(d) for d in json.loads(src.read_text(encoding="utf-8"))]
    manifest = {m["id"]: m for m in json.loads(manifest_path.read_text(encoding="utf-8"))}

    out_dir = run_dir / "fulltext"
    jsonl = out_dir / "eligibility.jsonl"

    # Si le protocole, les prompts ou la disponibilité des textes ont changé, les anciennes décisions sont archivées
    material = (
        json.dumps(protocol.model_dump(), sort_keys=True) + SYSTEM_REVIEWER + STANCE_A + STANCE_B + SYSTEM_ARBITER
        + json.dumps({k: v["status"] for k, v in sorted(manifest.items())})
    )
    signature = hashlib.sha256(material.encode()).hexdigest()[:12]
    sig_file = out_dir / "eligibility_signature.txt"
    if jsonl.exists() and (not sig_file.exists() or sig_file.read_text().strip() != signature):
        backup = out_dir / f"eligibility.{datetime.now():%Y%m%d_%H%M%S}.bak.jsonl"
        jsonl.rename(backup)
        print(f"  Protocole, prompts ou textes modifiés : anciennes décisions archivées dans {backup.name}")
    sig_file.write_text(signature)

    done: dict[str, ScreeningRecord] = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = ScreeningRecord.model_validate_json(line)
                done[rec.paper_id] = rec
        if done:
            print(f"  Reprise : {len(done)} articles déjà évalués.")

    llm = get_llm()
    t0, new = time.time(), 0
    for i, paper in enumerate(papers, 1):
        if paper.id in done:
            continue
        status = manifest.get(paper.id, {}).get("status", "unavailable")
        try:
            if status == "unavailable":
                rec = not_retrieved(paper)
            elif status == "full_text":
                from rag.doc_index import DocIndex  # import tardif : numpy/chromadb requis ici seulement

                text = (out_dir / "text" / f"{paper.id}.txt").read_text(encoding="utf-8")
                context = DocIndex(text).context(_queries(protocol))
                rec = assess_one(paper, protocol, llm, "full_text", context)
            else:
                rec = assess_one(paper, protocol, llm, "abstract_only", "")
        except Exception as e:  # noqa: BLE001  un échec isolé ne doit pas arrêter toute l'évaluation
            log.error("[%s] évaluation échouée : %s", paper.id, e)
            rec = ScreeningRecord(
                paper_id=paper.id, title=paper.title, stage="full_text", evidence_level=status,
                final_decision="maybe", final_rationale=f"Automatic assessment failed ({str(e)[:200]}); needs human review.",
                decided_by="error", needs_human_review=True,
            )
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")
        done[paper.id] = rec
        new += 1
        remaining = sum(1 for p in papers if p.id not in done)
        eta = (time.time() - t0) / new * remaining
        print(f"  [{i}/{len(papers)}] {paper.id} {rec.final_decision:<7} ({rec.evidence_level}, {rec.decided_by}, "
              f"{rec.final_criterion or '-'}) ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)

    overrides = _load_overrides(out_dir)
    unknown = sorted(set(overrides) - {p.id for p in papers})
    if unknown:
        print(f"  Attention : identifiants inconnus dans human_overrides.csv (ignorés) : {', '.join(unknown)}")
    records = [
        _apply_override(done[p.id], *overrides[p.id]) if p.id in overrides else _flag_exclusions(done[p.id])
        for p in papers if p.id in done
    ]
    final = {r.paper_id: r for r in records}
    summary = summarize(records, protocol)
    (out_dir / "prisma_eligibility.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out_dir / "eligibility_results.csv", records, protocol)

    keep = {r.paper_id for r in records if r.final_decision in ("include", "maybe") and r.final_criterion != "RULE:not_retrieved"}
    (out_dir / "included_for_extraction.json").write_text(
        json.dumps(
            [
                {**p.model_dump(), "evidence_level": final[p.id].evidence_level, "eligibility": final[p.id].final_decision}
                for p in papers if p.id in keep
            ],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def print_eligibility(s: dict) -> None:
    print("\n=== PRISMA : ÉLIGIBILITÉ SUR TEXTE INTÉGRAL ===")
    print(f"  Rapports recherchés        : {s['reports_sought']}")
    print(f"  Non récupérés              : {s['reports_not_retrieved']}")
    print(f"  Rapports évalués           : {s['reports_assessed']} "
          f"(texte intégral {s['assessed_on_full_text']}, résumé seul {s['assessed_on_abstract_only']})")
    print(f"  Exclus                     : {s['excluded']}")
    print(f"  Inclus                     : {s['included']} | À trancher par un humain : {s['maybe_human_decision']}")
    print(f"  Arbitrages : {s['arbiter_calls']} | À revoir : {s['needs_human_review']} | Erreurs : {s['errors']} | Kappa : {s['cohens_kappa']}")
    print(f"  Décisions humaines appliquées : {s['human_decisions']}")
    for reason, n in sorted(s["exclusion_reasons"].items(), key=lambda kv: -kv[1]):
        print(f"    - {n} x {reason}")
