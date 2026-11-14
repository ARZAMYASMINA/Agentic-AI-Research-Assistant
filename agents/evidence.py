"""Extraction de données + évaluation du risque de biais pour les articles retenus à l'éligibilité.

Sorties dans <run>/evidence/ : records.jsonl, evidence_table.csv, risk_of_bias.csv, prisma_evidence.json.
Reprenable article par article. Un index RAG est construit par article (une seule fois) et sert aux deux agents.
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

from agents import appraisal as appraisal_mod
from agents import extraction as extraction_mod
from agents.appraisal import TOOLS, abstract_only_results, appraise, domain_queries, overall_risk
from agents.extraction import DEFAULT_GROUPS, extract_group, field_queries, finalize_fields
from agents.llm import get_llm
from agents.schemas import EvidenceRecord, FieldValue, Protocol
from tools.paper import Paper
from tools.quotes import numbers_in_source, verify_quote

log = logging.getLogger(__name__)


def _header(paper: Paper) -> str:
    return f"Title: {paper.title}\nYear: {paper.year or 'unknown'}\nAbstract: {paper.abstract[:2500] or '(no abstract available)'}"


def _context(paper: Paper, index, queries: list[str]) -> str:
    if index is None:
        return _header(paper)
    return f"{_header(paper)}\n\nExcerpts from the full text:\n{index.context(queries, per_query=2, max_chars=8000)}"


def process_paper(paper: Paper, meta: dict, text: str, groups: list[dict], tool: str, llm) -> EvidenceRecord:
    level = meta.get("evidence_level", "none")
    specs = [f for g in groups for f in g["fields"]]
    rec = EvidenceRecord(
        paper_id=paper.id, title=paper.title, year=paper.year, evidence_level=level,
        eligibility=meta.get("eligibility", ""), tool=tool,
    )
    if level == "none":
        rec.extracted = finalize_fields([], specs, "")
        rec.risk_of_bias = abstract_only_results(tool, "No text or abstract available.")
        rec.error = "no text or abstract available"
        return rec

    index = None
    if level == "full_text":
        from rag.doc_index import DocIndex  # import tardif : numpy/chromadb requis ici seulement

        index = DocIndex(text)
    source = f"{paper.title}\n{paper.abstract}\n{text}"  # tout ce que le modèle a pu voir : sert à vérifier les citations

    extracted = {}
    for g in groups:
        batch = extract_group(paper, level, _context(paper, index, field_queries(g)), g, llm)
        extracted.update(finalize_fields(batch.entries, g["fields"], source))
    rec.extracted = extracted

    if level == "full_text":
        rec.risk_of_bias = appraise(paper, _context(paper, index, domain_queries(tool)), tool, source, llm)
    else:
        rec.risk_of_bias = abstract_only_results(tool)
    rec.overall_risk = overall_risk(rec.risk_of_bias)
    return rec


# --------------------------------------------------------------------------- rapports
def _flat(s: str) -> str:
    return " ".join(str(s).split())


def summarize(records: list[EvidenceRecord]) -> dict:
    ok = [r for r in records if not r.error]
    fields = [fv for r in ok for fv in r.extracted.values()]
    found = [fv for fv in fields if fv.found]
    per_field: dict[str, list[bool]] = {}
    for r in ok:
        for name, fv in r.extracted.items():
            per_field.setdefault(name, []).append(fv.found)
    return {
        "studies": len(records),
        "full_text": sum(r.evidence_level == "full_text" for r in records),
        "abstract_only": sum(r.evidence_level == "abstract_only" for r in records),
        "errors": sum(bool(r.error) for r in records),
        "fields_requested": len(fields),
        "fields_found": len(found),
        "found_rate": round(len(found) / len(fields), 3) if fields else None,
        "quotes_verified": sum(fv.evidence_verified for fv in found),
        "quote_verification_rate": round(sum(fv.evidence_verified for fv in found) / len(found), 3) if found else None,
        "grounded": sum(fv.grounded for fv in found),
        "grounded_rate": round(sum(fv.grounded for fv in found) / len(found), 3) if found else None,
        "field_found_rate": {n: round(sum(v) / len(v), 2) for n, v in per_field.items()},
        "overall_risk_of_bias": dict(Counter(r.overall_risk for r in records)),
    }


def _write_tables(out_dir: Path, records: list[EvidenceRecord], groups: list[dict], tool: str) -> None:
    names = [f["name"] for g in groups for f in g["fields"]]
    domains = list(TOOLS[tool])
    with (out_dir / "evidence_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "title", "evidence_level", "eligibility", *names, "unverified_fields", "overall_risk"])
        for r in records:
            unverified = [n for n, fv in r.extracted.items() if fv.found and not fv.grounded]
            w.writerow([
                r.paper_id, r.year or "", r.title, r.evidence_level, r.eligibility,
                *[_flat(r.extracted[n].value) if n in r.extracted else "" for n in names],
                ", ".join(unverified), r.overall_risk,
            ])
    with (out_dir / "risk_of_bias.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "evidence_level", *domains, "overall"])
        for r in records:
            by = {d.domain: d.judgement for d in r.risk_of_bias}
            w.writerow([r.paper_id, r.title, r.evidence_level, *[by.get(d, "unclear") for d in domains], r.overall_risk])


# --------------------------------------------------------------------------- pipeline
def _signature(protocol: Protocol, groups: list[dict], tool: str, metas: dict | None = None) -> str:
    """Empreinte de ce qui invalide TOUTES les extractions (protocole, champs, outil, prompts).

    La liste des études n'en fait pas partie : exclure une étude ne doit pas forcer la ré-extraction des autres.
    Un changement de niveau de preuve (PDF ajouté...) est traité article par article dans run_extraction.
    """
    material = (
        json.dumps(protocol.model_dump(), sort_keys=True) + json.dumps(groups, sort_keys=True) + tool
        + extraction_mod.SYSTEM + appraisal_mod.SYSTEM + json.dumps(TOOLS[tool], sort_keys=True)
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- décisions humaines
OVERRIDES_FILE = "human_overrides.csv"  # dans <run>/evidence/ : id,kind,name,value,note


def load_evidence_overrides(out_dir: Path) -> list[dict]:
    """Lit evidence/human_overrides.csv.

    kind = « field » : name = champ d'extraction, value = nouvelle valeur ;
    kind = « rob »   : name = domaine du risque de biais, value = low / high / unclear.
    """
    path = out_dir / OVERRIDES_FILE
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row = {k: (v or "").strip() for k, v in row.items() if k}
            if not row.get("id"):
                continue
            kind = row.get("kind", "").lower()
            if kind not in ("field", "rob") or not row.get("name") or not row.get("value"):
                log.warning("Ligne ignorée dans evidence/human_overrides.csv : %s", row)
                continue
            if kind == "rob" and row["value"].lower() not in ("low", "high", "unclear"):
                log.warning("Jugement inconnu (low/high/unclear attendu) : %s", row)
                continue
            rows.append({**row, "kind": kind})
    return rows


def apply_evidence_overrides(records: list[EvidenceRecord], overrides: list[dict]) -> tuple[list[EvidenceRecord], list[str]]:
    """Applique les corrections humaines PAR-DESSUS les résultats automatiques (records.jsonl n'est pas modifié).

    Renvoie les enregistrements corrigés et la liste des lignes inapplicables (identifiant, champ ou domaine inconnu).
    """
    by_id = {r.paper_id: r.model_copy(deep=True) for r in records}
    problems: list[str] = []
    for o in overrides:
        rec = by_id.get(o["id"])
        if rec is None:
            problems.append(f"{o['id']} : étude absente des études incluses")
            continue
        note = o.get("note") or "no note"
        if o["kind"] == "field":
            if o["name"] not in rec.extracted:
                problems.append(f"{o['id']} : champ inconnu « {o['name']} »")
                continue
            old = rec.extracted[o["name"]]
            rec.extracted[o["name"]] = FieldValue(
                value=o["value"], found=True, human=True,
                evidence=f"Human: {note} (automatic value was: {old.value[:200]})",
            )
        else:
            doms = {d.domain: i for i, d in enumerate(rec.risk_of_bias)}
            if o["name"] not in doms:
                problems.append(f"{o['id']} : domaine inconnu « {o['name']} »")
                continue
            i = doms[o["name"]]
            old = rec.risk_of_bias[i]
            rec.risk_of_bias[i] = old.model_copy(update={
                "judgement": o["value"].lower(), "human": True,
                "rationale": f"Human decision: {note} (automatic judgement was: {old.judgement} - {old.rationale[:200]})",
            })
            rec.overall_risk = overall_risk(rec.risk_of_bias)
    return [by_id[r.paper_id] for r in records], problems


def current_records(run_dir: Path) -> tuple[list[EvidenceRecord], list[str]]:
    """Études actuellement incluses (dans l'ordre de l'éligibilité), avec les décisions humaines appliquées."""
    jsonl = run_dir / "evidence" / "records.jsonl"
    src = run_dir / "fulltext" / "included_for_extraction.json"
    if not jsonl.exists():
        raise SystemExit("evidence/records.jsonl introuvable : lancez d'abord l'extraction (--extract-only).")
    metas = {d["id"]: d for d in json.loads(src.read_text(encoding="utf-8"))}
    done: dict[str, EvidenceRecord] = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = EvidenceRecord.model_validate_json(line)
            done[rec.paper_id] = rec
    problems = [f"{pid} : inclus mais pas encore extrait (relancez --extract-only)" for pid in metas if pid not in done]
    records = [
        done[pid].model_copy(update={"eligibility": meta.get("eligibility", done[pid].eligibility)})
        for pid, meta in metas.items() if pid in done
    ]
    records, more = apply_evidence_overrides(records, load_evidence_overrides(run_dir / "evidence"))
    return records, problems + more


def refresh_evidence_tables(run_dir: Path, cfg: dict) -> dict:
    """Réécrit evidence_table.csv, risk_of_bias.csv et prisma_evidence.json (études incluses + décisions humaines)."""
    groups = cfg.get("extraction", {}).get("groups") or DEFAULT_GROUPS
    tool = cfg.get("appraisal", {}).get("tool", "QUADAS-2")
    records, problems = current_records(run_dir)
    out_dir = run_dir / "evidence"
    summary = summarize(records)
    summary["human_overrides"] = sum(fv.human for r in records for fv in r.extracted.values()) + sum(
        d.human for r in records for d in r.risk_of_bias)
    summary["warnings"] = problems
    (out_dir / "prisma_evidence.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_tables(out_dir, records, groups, tool)
    return summary


def _reverify_fields(rec: EvidenceRecord, source: str) -> int:
    """Recalcule evidence_verified / numbers_verified des champs déjà extraits (code de vérification à jour)."""
    changed = 0
    for fv in rec.extracted.values():
        if not fv.found:
            continue
        parts = [p for p in fv.evidence.split(" | ") if p.strip()]
        ev = bool(parts) and all(verify_quote(p, source) for p in parts)
        nv = numbers_in_source(fv.value, source)
        if (ev, nv) != (fv.evidence_verified, fv.numbers_verified):
            fv.evidence_verified, fv.numbers_verified = ev, nv
            changed += 1
    return changed


def run_reappraisal(run_dir: Path, protocol: Protocol, cfg: dict) -> dict:
    """Refait le risque de biais (texte intégral) et re-vérifie les champs, SANS refaire l'extraction.

    Sert quand seuls le prompt d'évaluation ou le code de vérification ont changé. L'ancien records.jsonl est
    archivé ; la signature est mise à jour pour qu'un --extract-only ultérieur ne ré-extraie pas tout.
    """
    out_dir = run_dir / "evidence"
    jsonl = out_dir / "records.jsonl"
    src = run_dir / "fulltext" / "included_for_extraction.json"
    if not jsonl.exists() or not src.exists():
        raise SystemExit("evidence/records.jsonl introuvable : lancez d'abord l'extraction (--extract-only).")
    metas = {d["id"]: d for d in json.loads(src.read_text(encoding="utf-8"))}
    groups = cfg.get("extraction", {}).get("groups") or DEFAULT_GROUPS
    tool = cfg.get("appraisal", {}).get("tool", "QUADAS-2")
    records = [EvidenceRecord.model_validate_json(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    if any(r.tool != tool for r in records):
        raise SystemExit("L'outil d'évaluation a changé : relancez l'extraction complète (--extract-only).")

    backup = out_dir / f"records.{datetime.now():%Y%m%d_%H%M%S}.bak.jsonl"
    jsonl.rename(backup)
    print(f"  Anciens résultats archivés dans {backup.name}")

    llm = get_llm()
    t0 = time.time()
    with jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            meta = metas.get(r.paper_id)
            if meta is None or r.error:
                f.write(r.model_dump_json() + "\n")
                continue
            paper = Paper.model_validate(meta)
            text = ""
            if r.evidence_level == "full_text":
                text = (run_dir / "fulltext" / "text" / f"{paper.id}.txt").read_text(encoding="utf-8")
            source = f"{paper.title}\n{paper.abstract}\n{text}"
            changed = _reverify_fields(r, source)
            before = r.overall_risk
            if r.evidence_level == "full_text":
                from rag.doc_index import DocIndex

                index = DocIndex(text)
                r.risk_of_bias = appraise(paper, _context(paper, index, domain_queries(tool)), tool, source, llm)
                r.overall_risk = overall_risk(r.risk_of_bias)
            f.write(r.model_dump_json() + "\n")
            doms = "/".join(d.judgement for d in r.risk_of_bias)
            print(f"  {r.paper_id} {r.evidence_level:<13} champs re-vérifiés modifiés {changed}  "
                  f"biais {before} -> {r.overall_risk} ({doms})", flush=True)
    print(f"  Durée : {time.time() - t0:.0f} s")

    (out_dir / "signature.txt").write_text(_signature(protocol, groups, tool))
    return refresh_evidence_tables(run_dir, cfg)


def run_extraction(run_dir: Path, protocol: Protocol, cfg: dict, limit: int | None = None) -> dict:
    src = run_dir / "fulltext" / "included_for_extraction.json"
    if not src.exists():
        raise SystemExit("fulltext/included_for_extraction.json introuvable : lancez d'abord l'éligibilité (--eligibility-only).")
    metas = {d["id"]: d for d in json.loads(src.read_text(encoding="utf-8"))}
    papers = [Paper.model_validate(d) for d in metas.values()]
    if limit:
        papers = papers[:limit]

    groups = cfg.get("extraction", {}).get("groups") or DEFAULT_GROUPS
    tool = cfg.get("appraisal", {}).get("tool", "QUADAS-2")
    if tool not in TOOLS:
        raise SystemExit(f"Outil d'évaluation inconnu : '{tool}'. Disponibles : {', '.join(TOOLS)} (config appraisal.tool).")

    out_dir = run_dir / "evidence"
    out_dir.mkdir(exist_ok=True)
    jsonl = out_dir / "records.jsonl"

    # Si le protocole, les champs, l'outil ou les prompts ont changé, les anciens résultats sont archivés
    signature = _signature(protocol, groups, tool)
    sig_file = out_dir / "signature.txt"
    if jsonl.exists() and (not sig_file.exists() or sig_file.read_text().strip() != signature):
        backup = out_dir / f"records.{datetime.now():%Y%m%d_%H%M%S}.bak.jsonl"
        jsonl.rename(backup)
        print(f"  Protocole, champs, outil ou prompts modifiés : anciens résultats archivés dans {backup.name}")
    sig_file.write_text(signature)

    done: dict[str, EvidenceRecord] = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = EvidenceRecord.model_validate_json(line)
                done[rec.paper_id] = rec
        if done:
            print(f"  Reprise : {len(done)} articles déjà traités.")

    llm = get_llm()
    t0, new = time.time(), 0
    for i, paper in enumerate(papers, 1):
        if paper.id in done and done[paper.id].evidence_level == metas[paper.id].get("evidence_level"):
            continue  # déjà traité avec le même niveau de preuve (un PDF ajouté entre-temps force le retraitement)
        meta = metas[paper.id]
        try:
            text = ""
            if meta.get("evidence_level") == "full_text":
                text = (run_dir / "fulltext" / "text" / f"{paper.id}.txt").read_text(encoding="utf-8")
            rec = process_paper(paper, meta, text, groups, tool, llm)
        except Exception as e:  # noqa: BLE001  un échec isolé ne doit pas arrêter toute l'extraction
            log.error("[%s] extraction échouée : %s", paper.id, e)
            rec = EvidenceRecord(
                paper_id=paper.id, title=paper.title, year=paper.year, evidence_level=meta.get("evidence_level", ""),
                eligibility=meta.get("eligibility", ""), tool=tool, error=f"extraction failed: {str(e)[:200]}",
            )
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")
        done[paper.id] = rec
        new += 1
        remaining = sum(1 for p in papers if p.id not in done)
        eta = (time.time() - t0) / new * remaining
        found = sum(fv.found for fv in rec.extracted.values())
        print(f"  [{i}/{len(papers)}] {paper.id} {rec.evidence_level:<13} champs {found}/{len(rec.extracted)} "
              f"biais {rec.overall_risk:<7} ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)

    return refresh_evidence_tables(run_dir, cfg)


def print_evidence(s: dict) -> None:
    print("\n=== EXTRACTION ET RISQUE DE BIAIS ===")
    print(f"  Études traitées         : {s['studies']} (texte intégral {s['full_text']}, résumé seul {s['abstract_only']}, erreurs {s['errors']})")
    print(f"  Champs renseignés       : {s['fields_found']}/{s['fields_requested']} ({s['found_rate']})")
    print(f"  Citations exactes       : {s['quotes_verified']}/{s['fields_found']} ({s['quote_verification_rate']})")
    print(f"  Valeurs étayées        : {s['grounded']}/{s['fields_found']} ({s['grounded_rate']}) (citation exacte ou chiffres retrouvés)")
    print(f"  Risque de biais global  : {s['overall_risk_of_bias']}")
    rates = ", ".join(f"{n} {r}" for n, r in s["field_found_rate"].items())
    print(f"  Taux par champ          : {rates}")
    if s.get("human_overrides"):
        print(f"  Corrections humaines    : {s['human_overrides']} (evidence/human_overrides.csv)")
    for w in s.get("warnings", []):
        print(f"  ATTENTION : {w}")
