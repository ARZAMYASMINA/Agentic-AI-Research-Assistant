import csv
import json

from agents.evaluation import (
    binary_metrics, evaluate_extraction, evaluate_reproducibility, evaluate_screening, prepare_screening_gold,
)


def _decisions(run, finals):
    (run / "screening").mkdir(exist_ok=True)
    rows = [{"paper_id": pid, "final_decision": d, "decided_by": "consensus"} for pid, d in finals.items()]
    (run / "screening" / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_binary_metrics_and_wss():
    agent = [True, True, True, False, False, False, False, False, False, False]
    truth = [True, True, False, True, False, False, False, False, False, False]
    m = binary_metrics(agent, truth)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 6)
    assert m["recall"] == 0.667 and m["precision"] == 0.667 and m["specificity"] == 0.857
    assert m["wss_at_recall"] == round(7 / 10 - (1 - 2 / 3), 3)


def test_gold_sample_is_blind_and_never_overwritten(tmp_path):
    finals = {f"P{i:04d}": ("include" if i < 5 else "exclude") for i in range(40)}
    _decisions(tmp_path, finals)
    (tmp_path / "records_deduplicated.json").write_text(json.dumps([{"id": k, "title": "t", "abstract": "a"} for k in finals]))
    path = prepare_screening_gold(tmp_path, n=20)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    ids = {r["id"] for r in rows}
    assert len(rows) == 20 and {f"P{i:04d}" for i in range(5)} <= ids  # tous les retenus par l'agent
    assert "final_decision" not in rows[0] and "agent" not in " ".join(rows[0])  # aveugle
    path.write_text("filled", encoding="utf-8")
    assert prepare_screening_gold(tmp_path, n=20).read_text(encoding="utf-8") == "filled"


def test_evaluate_screening_uses_consensus_then_agreement(tmp_path):
    _decisions(tmp_path, {"P1": "include", "P2": "exclude", "P3": "maybe", "P4": "exclude", "P5": "exclude"})
    (tmp_path / "evaluation").mkdir()
    with (tmp_path / "evaluation" / "screening_gold.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "title", "abstract", "reviewer1", "reviewer2", "consensus", "note"])
        w.writerow(["P1", "", "", "", "include", "include", "", ""])
        w.writerow(["P2", "", "", "", "include", "maybe", "", ""])       # accord (positif) -> agent l'a raté
        w.writerow(["P3", "", "", "", "exclude", "include", "exclude", ""])  # consensus
        w.writerow(["P4", "", "", "", "exclude", "include", "", ""])     # désaccord sans consensus : ignoré
        w.writerow(["P5", "", "", "", "", "", "", ""])                   # non rempli
    r = evaluate_screening(tmp_path)
    assert r["labelled"] == 3 and r["missed_by_agent"] == ["P2"]
    m = r["agent_vs_human"]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 0)
    assert r["human_pairs"] == 4


def test_evaluate_extraction(tmp_path):
    (tmp_path / "evaluation").mkdir()
    with (tmp_path / "evaluation" / "extraction_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "field", "agent_value", "verdict", "correct_value", "note"])
        for v in ("correct", "correct", "partial", "incorrect", "na", ""):
            w.writerow(["P1", "auc", "x", v, "", ""])
    r = evaluate_extraction(tmp_path)
    assert r["audited"] == 5
    assert r["overall"]["accuracy_strict"] == 0.5 and r["overall"]["accuracy_lenient"] == 0.75


def test_reproducibility_summary(tmp_path):
    (tmp_path / "evaluation").mkdir()
    rows = [{"paper_id": "P1", "original": "include", "rerun": "include", "same_rationale_a": True},
            {"paper_id": "P2", "original": "exclude", "rerun": "maybe", "same_rationale_a": False},
            {"paper_id": "P3", "original": "exclude", "rerun": "exclude", "same_rationale_a": True}]
    (tmp_path / "evaluation" / "reproducibility.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    r = evaluate_reproducibility(tmp_path)
    assert r["identical_final_decision"] == 2 and r["identical_rationale_reviewer_a"] == 2
    assert r["changed"] == ["P2: exclude -> maybe"]


def test_human_audit_separates_overturned_resolved_confirmed(tmp_path):
    from agents.evaluation import human_audit
    from agents.human import append_override

    (tmp_path / "fulltext").mkdir()
    (tmp_path / "screening").mkdir()
    (tmp_path / "evidence").mkdir()
    rows = [{"paper_id": "P1", "final_decision": "include"}, {"paper_id": "P2", "final_decision": "maybe"},
            {"paper_id": "P3", "final_decision": "exclude"}]
    (tmp_path / "fulltext" / "eligibility.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "fulltext" / "included_for_extraction.json").write_text("[]")
    for pid, d in (("P1", "exclude"), ("P2", "include"), ("P3", "exclude")):
        append_override(tmp_path / "fulltext", pid, d, "n")
    e = human_audit(tmp_path)["eligibility"]
    assert (e["reviewed"], e["overturned"], e["maybe_resolved"], e["confirmed"]) == (3, 1, 1, 1)
    assert e["overturned_ids"] == ["P1"]
