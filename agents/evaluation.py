"""Évaluation du pipeline (étape 8) : exactitude, cohérence, reproductibilité, gain de temps.

1. Screening contre une vérité terrain humaine : gabarit à remplir en aveugle par deux membres de l'équipe
   (evaluation/screening_gold.csv), puis rappel, précision, F1, spécificité, WSS, kappa humain-humain et agent-humain.
2. Extraction : audit humain des valeurs extraites (evaluation/extraction_audit.csv : correct / partial / incorrect
   / missed), exactitude par champ.
3. Audit des décisions déjà relues par des humains (décisions automatiques changées / confirmées).
4. Reproductibilité : nouveau screening d'un échantillon avec les mêmes réglages (evaluation/reproducibility.jsonl).
5. Gain de temps : temps machine mesuré dans trace.jsonl contre un temps manuel fondé sur des hypothèses explicites
   (configurables dans protocol_template.yaml, section `evaluation.manual_minutes`).
Tout est écrit dans evaluation/evaluation_report.md et evaluation/evaluation.json.
"""
from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tools.metrics import cohen_kappa

# Hypothèses de temps manuel (minutes), à citer comme hypothèses dans le rapport ; modifiables dans la config.
DEFAULT_MANUAL_MINUTES = {
    "screening_per_record_per_reviewer": 1.0,   # lecture titre + résumé et décision
    "fulltext_eligibility_per_report_per_reviewer": 10.0,
    "extraction_per_study": 60.0,
    "risk_of_bias_per_study": 30.0,
}
POSITIVE = {"include", "maybe"}  # au titre/résumé, « maybe » passe au texte intégral : compté comme positif


def _eval_dir(run_dir: Path) -> Path:
    d = run_dir / "evaluation"
    d.mkdir(exist_ok=True)
    return d


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _last_by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["paper_id"]: r for r in rows}


# --------------------------------------------------------------------------- 1. gabarits de vérité terrain
def prepare_screening_gold(run_dir: Path, n: int = 100, seed: int = 2026) -> Path:
    """Échantillon aléatoire stratifié (tous les articles retenus par l'agent + exclus tirés au hasard), SANS les
    décisions de l'agent (aveugle). L'ordre est mélangé pour qu'on ne devine pas la décision de l'agent."""
    out = _eval_dir(run_dir) / "screening_gold.csv"
    if out.exists():
        return out  # ne jamais écraser un fichier peut-être déjà rempli
    papers = {d["id"]: d for d in json.loads((run_dir / "records_deduplicated.json").read_text(encoding="utf-8"))}
    decisions = _last_by_id(_jsonl(run_dir / "screening" / "decisions.jsonl"))
    rng = random.Random(seed)
    passed = sorted(i for i, r in decisions.items() if r["final_decision"] in POSITIVE)
    excluded = sorted(i for i, r in decisions.items() if r["final_decision"] not in POSITIVE)
    k = max(0, min(len(excluded), n - len(passed)))
    sample = passed + rng.sample(excluded, k)
    rng.shuffle(sample)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "title", "abstract", "reviewer1", "reviewer2", "consensus", "note"])
        for pid in sample:
            p = papers.get(pid, {})
            w.writerow([pid, p.get("year") or "", p.get("title", ""), (p.get("abstract") or "(no abstract)")[:3000], "", "", "", ""])
    return out


def prepare_extraction_audit(run_dir: Path) -> Path:
    """Valeurs extraites des études en texte intégral, à juger par un humain (correct / partial / incorrect / missed)."""
    from agents.evidence import current_records

    out = _eval_dir(run_dir) / "extraction_audit.csv"
    if out.exists():
        return out
    records, _ = current_records(run_dir)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "field", "agent_value", "verdict", "correct_value", "note"])
        for r in records:
            if r.evidence_level != "full_text":
                continue
            for name, fv in r.extracted.items():
                if fv.human:
                    continue  # déjà corrigé par un humain : ne pas l'auditer comme une sortie de l'agent
                w.writerow([r.paper_id, name, " ".join(fv.value.split()), "", "", ""])
    return out


def instructions_md(run_dir: Path) -> Path:
    from agents.schemas import Protocol

    p = Protocol.model_validate_json((run_dir / "protocol.json").read_text(encoding="utf-8"))
    crit = "\n".join(f"- I{i}. {c}" for i, c in enumerate(p.inclusion_criteria, 1))
    crit += "\n" + "\n".join(f"- E{i}. {c}" for i, c in enumerate(p.exclusion_criteria, 1))
    text = f"""# Instructions pour la vérité terrain (étape 8)

## A. Screening titre/résumé — `screening_gold.csv`
Deux membres de l'équipe remplissent **chacun de leur côté, sans se concerter et sans regarder les résultats de
l'agent**, la colonne `reviewer1` (puis `reviewer2`) avec `include`, `exclude` ou `maybe` (« maybe » = à lire en texte
intégral). Ensuite, pour chaque désaccord, remplir `consensus` après discussion (sinon la laisser vide).
Noter le **temps passé** (début/fin) : il sert à l'évaluation du gain de temps.

Question : {p.research_question}

Critères :
{crit}

## B. Extraction — `extraction_audit.csv`
Pour chaque ligne (étude en texte intégral, champ, valeur de l'agent), ouvrir le texte
(`fulltext/text/<id>.txt` ou le PDF) et remplir `verdict` :
`correct` (valeur juste et complète), `partial` (juste mais incomplète ou imprécise), `incorrect` (fausse ou attribuée
au mauvais groupe), `missed` (l'agent dit « not reported » alors que l'information existe) ou `na` (réellement non
rapporté : l'agent a raison de dire « not reported »). Si besoin, écrire la bonne valeur dans `correct_value`.

## C. Calcul
`python main.py --resume <RUN> --eval` recalcule `evaluation_report.md` à chaque fois (les lignes vides sont ignorées).
"""
    out = _eval_dir(run_dir) / "INSTRUCTIONS.md"
    out.write_text(text, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- 2. métriques
def binary_metrics(agent: list[bool], truth: list[bool]) -> dict:
    tp = sum(a and t for a, t in zip(agent, truth))
    fp = sum(a and not t for a, t in zip(agent, truth))
    fn = sum((not a) and t for a, t in zip(agent, truth))
    tn = sum((not a) and (not t) for a, t in zip(agent, truth))
    n = len(agent)
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    spec = tn / (tn + fp) if tn + fp else None
    # Work Saved over Sampling au rappel atteint (classement binaire) : (TN + FN) / N - (1 - rappel)
    wss = (tn + fn) / n - (1 - recall) if n and recall is not None else None
    k = cohen_kappa(agent, truth)
    r = lambda x: round(x, 3) if x is not None else None  # noqa: E731
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": r(recall), "precision": r(precision),
            "f1": r(f1), "specificity": r(spec), "wss_at_recall": r(wss), "kappa": r(k)}


def evaluate_screening(run_dir: Path) -> dict | None:
    path = run_dir / "evaluation" / "screening_gold.csv"
    if not path.exists():
        return None
    decisions = _last_by_id(_jsonl(run_dir / "screening" / "decisions.jsonl"))
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    norm = lambda s: (s or "").strip().lower()  # noqa: E731
    pairs, h1h2, missed = [], ([], []), []
    for row in rows:
        r1, r2, c = norm(row.get("reviewer1")), norm(row.get("reviewer2")), norm(row.get("consensus"))
        valid = {"include", "exclude", "maybe"}
        if r1 in valid and r2 in valid:
            h1h2[0].append(r1 in POSITIVE)
            h1h2[1].append(r2 in POSITIVE)
        truth = c if c in valid else (r1 if r1 in valid and (r2 not in valid or (r1 in POSITIVE) == (r2 in POSITIVE)) else None)
        if truth is None or row["id"] not in decisions:
            continue
        agent_pass = decisions[row["id"]]["final_decision"] in POSITIVE
        pairs.append((agent_pass, truth in POSITIVE))
        if truth in POSITIVE and not agent_pass:
            missed.append(row["id"])
    res = {"labelled": len(pairs), "rows": len(rows), "missed_by_agent": missed}
    if pairs:
        res["agent_vs_human"] = binary_metrics([a for a, _ in pairs], [t for _, t in pairs])
    if h1h2[0]:
        k = cohen_kappa(*h1h2)
        res["human_vs_human_kappa"] = round(k, 3) if k is not None else None
        res["human_pairs"] = len(h1h2[0])
    return res


def evaluate_extraction(run_dir: Path) -> dict | None:
    path = run_dir / "evaluation" / "extraction_audit.csv"
    if not path.exists():
        return None
    per: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in csv.DictReader(path.open(encoding="utf-8-sig", newline="")):
        v = (row.get("verdict") or "").strip().lower()
        if v in ("correct", "partial", "incorrect", "missed", "na"):
            per[row["field"]][v] += 1
    if not per:
        return {"audited": 0}
    tot = defaultdict(int)
    for f in per.values():
        for k, v in f.items():
            tot[k] += v
    scored = lambda d: sum(d[k] for k in ("correct", "partial", "incorrect", "missed"))  # noqa: E731
    acc = lambda d: round(d["correct"] / scored(d), 3) if scored(d) else None  # noqa: E731
    return {
        "audited": sum(tot.values()),
        "overall": {**tot, "accuracy_strict": acc(tot),
                    "accuracy_lenient": round((tot["correct"] + tot["partial"]) / scored(tot), 3) if scored(tot) else None},
        "per_field": {k: {**v, "accuracy_strict": acc(v)} for k, v in sorted(per.items())},
    }


# --------------------------------------------------------------------------- 3. audit des décisions relues
def human_audit(run_dir: Path) -> dict:
    """Ce que les humains ont changé parmi les décisions automatiques qu'ils ont relues."""
    from agents.human import load_overrides

    out = {}
    for stage, folder, name in (("screening", "screening", "decisions.jsonl"), ("eligibility", "fulltext", "eligibility.jsonl")):
        auto = _last_by_id(_jsonl(run_dir / folder / name))
        ov = load_overrides(run_dir / folder)
        known = {pid: d for pid, (d, _, _) in ov.items() if pid in auto}
        overturned = [p for p, d in known.items() if auto[p]["final_decision"] in ("include", "exclude") and auto[p]["final_decision"] != d]
        resolved = [p for p, d in known.items() if auto[p]["final_decision"] == "maybe" and d != "maybe"]
        confirmed = [p for p, d in known.items() if auto[p]["final_decision"] == d]
        out[stage] = {"reviewed": len(known), "overturned": len(overturned), "maybe_resolved": len(resolved),
                      "confirmed": len(confirmed), "overturned_ids": overturned, "resolved_ids": resolved}
    ev = run_dir / "evidence" / "human_overrides.csv"
    fields = rob = 0
    if ev.exists():
        for row in csv.DictReader(ev.open(encoding="utf-8-sig", newline="")):
            fields += row.get("kind") == "field"
            rob += row.get("kind") == "rob"
    recs = _jsonl(run_dir / "evidence" / "records.jsonl")
    included = {d["id"] for d in json.loads((run_dir / "fulltext" / "included_for_extraction.json").read_text(encoding="utf-8"))}
    ft = [r for r in recs if r["paper_id"] in included and r["evidence_level"] == "full_text"]
    out["risk_of_bias"] = {"domains_assessed_full_text": sum(len(r["risk_of_bias"]) for r in ft), "domains_corrected": rob}
    out["extraction"] = {"values_corrected": fields,
                         "values_extracted": sum(1 for r in recs if r["paper_id"] in included for v in r["extracted"].values() if v["found"])}
    return out


# --------------------------------------------------------------------------- 4. reproductibilité
def run_reproducibility(run_dir: Path, n: int = 30, seed: int = 7) -> dict:
    """Refait le screening (A, B, arbitre) d'un échantillon avec les mêmes réglages et compare aux décisions d'origine."""
    from agents.llm import get_llm
    from agents.schemas import Protocol, ScreeningRecord
    from agents.screening import _load_papers, screen_one

    protocol = Protocol.model_validate_json((run_dir / "protocol.json").read_text(encoding="utf-8"))
    original = _last_by_id(_jsonl(run_dir / "screening" / "decisions.jsonl"))
    papers = {p.id: p for p in _load_papers(run_dir)}
    ids = sorted(i for i, r in original.items() if r["decided_by"] in ("consensus", "arbiter"))
    sample = sorted(random.Random(seed).sample(ids, min(n, len(ids))))
    path = _eval_dir(run_dir) / "reproducibility.jsonl"
    done = _last_by_id(_jsonl(path))
    llm = get_llm()
    for k, pid in enumerate(sample, 1):
        if pid in done:
            continue
        t0 = time.time()
        rec: ScreeningRecord = screen_one(papers[pid], protocol, llm)
        row = {"paper_id": pid, "original": original[pid]["final_decision"], "rerun": rec.final_decision,
               "original_a": (original[pid].get("reviewer_a") or {}).get("decision"),
               "rerun_a": rec.reviewer_a.decision if rec.reviewer_a else None,
               "same_rationale_a": bool(rec.reviewer_a and original[pid].get("reviewer_a")
                                        and rec.reviewer_a.rationale == original[pid]["reviewer_a"]["rationale"]),
               "seconds": round(time.time() - t0, 1)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        done[pid] = row
        print(f"  [{k}/{len(sample)}] {pid} {row['original']} -> {row['rerun']}", flush=True)
    return evaluate_reproducibility(run_dir)


def evaluate_reproducibility(run_dir: Path) -> dict | None:
    rows = _jsonl(run_dir / "evaluation" / "reproducibility.jsonl")
    if not rows:
        return None
    same = sum(r["original"] == r["rerun"] for r in rows)
    a = [r["original"] in POSITIVE for r in rows]
    b = [r["rerun"] in POSITIVE for r in rows]
    k = cohen_kappa(a, b)
    return {"n": len(rows), "identical_final_decision": same, "identical_rate": round(same / len(rows), 3),
            "identical_rationale_reviewer_a": sum(r["same_rationale_a"] for r in rows),
            "kappa_pass_exclude": round(k, 3) if k is not None else None,
            "changed": [f"{r['paper_id']}: {r['original']} -> {r['rerun']}" for r in rows if r["original"] != r["rerun"]]}


# --------------------------------------------------------------------------- 5. gain de temps
def time_savings(run_dir: Path, minutes: dict | None = None) -> dict:
    """Temps machine pour UN passage (temps moyen par appel mesuré × appels nécessaires) contre temps manuel estimé."""
    from agents.observability import trace_summary

    m = {**DEFAULT_MANUAL_MINUTES, **(minutes or {})}
    agents = trace_summary(run_dir)["agents"]
    mean = {k: v["seconds"] / v["calls"] for k, v in agents.items() if v["calls"]}
    scr = _jsonl(run_dir / "screening" / "decisions.jsonl")
    elig = _jsonl(run_dir / "fulltext" / "eligibility.jsonl")
    recs = _jsonl(run_dir / "evidence" / "records.jsonl")
    n_scr = sum(1 for r in scr if r.get("reviewer_a"))
    n_arb = sum(1 for r in scr if r.get("arbiter"))
    n_el = sum(1 for r in elig if r.get("reviewer_a"))
    n_el_arb = sum(1 for r in elig if r.get("arbiter"))
    n_ext = len(recs)
    n_ft = sum(r["evidence_level"] == "full_text" for r in recs)
    machine = {
        "screening": n_scr * (mean.get("screening_a", 0) + mean.get("screening_b", 0)) + n_arb * mean.get("screening_arbiter", 0),
        "fulltext_eligibility": (n_el * 2 + n_el_arb) * mean.get("eligibility", 0),
        "extraction": n_ext * 2 * mean.get("extraction", 0),  # deux groupes de champs par étude
        "risk_of_bias": n_ft * mean.get("appraisal", 0),
        "synthesis_swim": len({r["paper_id"] for r in recs}) * mean.get("swim", 0),
    }
    manual = {
        "screening": len(scr) * 2 * m["screening_per_record_per_reviewer"] * 60,
        "fulltext_eligibility": n_el * 2 * m["fulltext_eligibility_per_report_per_reviewer"] * 60,
        "extraction": n_ext * m["extraction_per_study"] * 60,
        "risk_of_bias": n_ft * m["risk_of_bias_per_study"] * 60,
        "synthesis_swim": None,
    }
    rows = {k: {"machine_min": round(machine[k] / 60, 1),
                "manual_min": round(manual[k] / 60, 1) if manual[k] is not None else None} for k in machine}
    tm = sum(v["machine_min"] for v in rows.values())
    th = sum(v["manual_min"] or 0 for v in rows.values())
    return {"assumptions_minutes": m, "counts": {"screened": len(scr), "arbiter": n_arb, "eligibility_assessed": n_el,
            "extracted": n_ext, "full_text": n_ft}, "stages": rows,
            "machine_total_min": round(tm, 1), "manual_total_min": round(th, 1),
            "ratio": round(th / tm, 1) if tm else None}


# --------------------------------------------------------------------------- rapport
def _fmt(x) -> str:
    return "n/a" if x is None else str(x)


def evaluation_report(run_dir: Path, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    s = json.loads((run_dir / "screening" / "prisma_screening.json").read_text(encoding="utf-8"))
    e = json.loads((run_dir / "fulltext" / "prisma_eligibility.json").read_text(encoding="utf-8"))
    res = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "screening_gold": evaluate_screening(run_dir),
        "extraction_audit": evaluate_extraction(run_dir),
        "human_audit": human_audit(run_dir),
        "reproducibility": evaluate_reproducibility(run_dir),
        "consistency": {"screening_kappa_A_B": s.get("cohens_kappa"), "screening_agreement_A_B": s.get("inter_reviewer_agreement"),
                        "eligibility_kappa_A_B": e.get("cohens_kappa")},
        "time": time_savings(run_dir, cfg.get("evaluation", {}).get("manual_minutes")),
    }
    swim = run_dir / "report" / "swim.json"
    if swim.exists():
        sw = json.loads(swim.read_text(encoding="utf-8"))
        res["swim"] = {"accepted": sum(len(v.get("valid", [])) for v in sw.values()),
                       "rejected": sum(len(v.get("rejected", [])) for v in sw.values())}
    out = _eval_dir(run_dir)
    (out / "evaluation.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "evaluation_report.md").write_text(report_md(res), encoding="utf-8")
    res["path"] = str(out / "evaluation_report.md")
    return res


def report_md(r: dict) -> str:
    L = ["# Evaluation report", "", f"*Generated {r['generated']}.*", ""]
    L += ["## 1. Screening accuracy against a human reference", ""]
    g = r["screening_gold"]
    if not g or not g.get("labelled"):
        L += ["_Not available yet: `evaluation/screening_gold.csv` has not been filled by the team (see INSTRUCTIONS.md)._", ""]
    else:
        m = g["agent_vs_human"]
        L += [f"{g['labelled']} records with a human reference (of {g['rows']} in the sample). Positive = include or maybe (passes to full text).", "",
              "| Recall | Precision | F1 | Specificity | WSS at achieved recall | Kappa agent-human | Kappa human-human |",
              "|---|---|---|---|---|---|---|",
              f"| {_fmt(m['recall'])} | {_fmt(m['precision'])} | {_fmt(m['f1'])} | {_fmt(m['specificity'])} | {_fmt(m['wss_at_recall'])} | {_fmt(m['kappa'])} | {_fmt(g.get('human_vs_human_kappa'))} |",
              "", f"Confusion: TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, TN {m['tn']}. Missed by the agent: {', '.join(g['missed_by_agent']) or 'none'}.",
              "The sample contains every record the agent passed plus random exclusions, so precision is representative but recall is "
              "estimated only from the excluded records that were sampled.", ""]
    L += ["## 2. Extraction accuracy (human audit of full-text extractions)", ""]
    x = r["extraction_audit"]
    if not x or not x.get("audited"):
        L += ["_Not available yet: `evaluation/extraction_audit.csv` has not been filled._", ""]
    else:
        o = x["overall"]
        L += [f"{x['audited']} values audited. Strict accuracy (correct only): {_fmt(o['accuracy_strict'])}; lenient (correct + partial): {_fmt(o['accuracy_lenient'])}.", "",
              "| Field | correct | partial | incorrect | missed | na | strict accuracy |", "|---|---|---|---|---|---|---|"]
        for f, v in x["per_field"].items():
            L.append(f"| {f} | {v.get('correct', 0)} | {v.get('partial', 0)} | {v.get('incorrect', 0)} | {v.get('missed', 0)} | {v.get('na', 0)} | {_fmt(v['accuracy_strict'])} |")
        L.append("")
    h = r["human_audit"]
    L += ["## 3. Human review of automatic decisions (audit trail)", "",
          "| Stage | Reviewed by a human | Overturned (include ↔ exclude) | 'Maybe' resolved | Confirmed |", "|---|---|---|---|---|"]
    for st in ("screening", "eligibility"):
        x = h[st]
        ids = f" ({', '.join(x['overturned_ids'])})" if x["overturned_ids"] else ""
        L.append(f"| {st} | {x['reviewed']} | {x['overturned']}{ids} | {x['maybe_resolved']} | {x['confirmed']} |")
    L += ["", f"Risk of bias: {h['risk_of_bias']['domains_corrected']} of {h['risk_of_bias']['domains_assessed_full_text']} "
          f"full-text domain judgements corrected by a human. Extraction: {h['extraction']['values_corrected']} of "
          f"{h['extraction']['values_extracted']} extracted values corrected. These figures come from targeted reviews "
          "(not random samples) and are not accuracy estimates.", ""]
    if r.get("swim"):
        L += [f"SWiM synthesis: {r['swim']['accepted']} comparisons accepted by the automatic checks, {r['swim']['rejected']} rejected.", ""]
    c = r["consistency"]
    L += ["## 4. Consistency between agents and reproducibility", "",
          f"- Simulated reviewers A and B (same model, different instructions): screening agreement {_fmt(c['screening_agreement_A_B'])}, "
          f"kappa {_fmt(c['screening_kappa_A_B'])}; full-text eligibility kappa {_fmt(c['eligibility_kappa_A_B'])}."]
    rp = r["reproducibility"]
    if rp:
        L.append(f"- Re-run of the title/abstract screening on {rp['n']} records with identical settings: {rp['identical_final_decision']}/{rp['n']} "
                 f"identical final decisions ({rp['identical_rate']}), kappa {_fmt(rp['kappa_pass_exclude'])}; reviewer A's rationale "
                 f"identical word for word in {rp['identical_rationale_reviewer_a']}/{rp['n']}. Changed: {', '.join(rp['changed']) or 'none'}.")
    else:
        L.append("- Reproducibility: not run yet (`python main.py --resume <RUN> --eval-repro 30`).")
    t = r["time"]
    L += ["", "## 5. Time", "",
          "Machine time = mean measured duration of each LLM call (trace.jsonl) × number of calls needed for one pass. "
          "Manual time = **assumptions** (minutes): " + ", ".join(f"{k} {v}" for k, v in t["assumptions_minutes"].items()) + ".", "",
          "| Stage | Machine (min) | Manual estimate (min) |", "|---|---|---|"]
    for k, v in t["stages"].items():
        L.append(f"| {k} | {v['machine_min']} | {_fmt(v['manual_min'])} |")
    L += [f"| **total** | **{t['machine_total_min']}** | **{t['manual_total_min']}** |", "",
          f"Ratio manual / machine ≈ {_fmt(t['ratio'])}. Human time still required (protocol approval, reviews of flagged decisions, "
          "corrections) is not included and should be measured by the team.", ""]
    return "\n".join(L)


def print_evaluation(res: dict) -> None:
    print("\n=== ÉVALUATION ===")
    g = res["screening_gold"]
    print(f"  Screening vs humain    : {g['agent_vs_human'] if g and g.get('labelled') else 'vérité terrain non remplie'}")
    x = res["extraction_audit"]
    print(f"  Audit d'extraction     : {x['overall'] if x and x.get('audited') else 'non rempli'}")
    print(f"  Reproductibilité       : {res['reproducibility'] or 'non lancée'}")
    t = res["time"]
    print(f"  Temps machine / manuel : {t['machine_total_min']} min / {t['manual_total_min']} min (ratio {t['ratio']})")
    print(f"  Rapport : {res['path']}")
