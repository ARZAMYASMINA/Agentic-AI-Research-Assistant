import json

from agents.human import append_override, load_overrides
from agents.review import interactive_review, review_queue
from agents.schemas import ScreeningDecision, ScreeningRecord


def _rec(pid, final, review=True, a="include", b="exclude"):
    return ScreeningRecord(
        paper_id=pid, title=f"Title {pid}", final_decision=final, final_criterion="E2" if final == "exclude" else "",
        final_rationale="auto", decided_by="arbiter", needs_human_review=review,
        reviewer_a=ScreeningDecision(rationale="ra", criterion="none", decision=a, confidence="low"),
        reviewer_b=ScreeningDecision(rationale="rb", criterion="E2", decision=b, confidence="low"),
    )


def _setup(tmp_path):
    (tmp_path / "screening").mkdir()
    recs = [_rec("P1", "exclude"), _rec("P2", "maybe"), _rec("P3", "include", review=False), _rec("P4", "exclude")]
    (tmp_path / "screening" / "decisions.jsonl").write_text("\n".join(r.model_dump_json() for r in recs), encoding="utf-8")
    (tmp_path / "records_deduplicated.json").write_text(json.dumps([{"id": "P1", "title": "t", "abstract": "abs"}]), encoding="utf-8")
    return tmp_path


def test_queue_skips_reviewed_and_not_flagged(tmp_path):
    run = _setup(tmp_path)
    assert [r.paper_id for r in review_queue(run, "screening")] == ["P1", "P2", "P4"]
    append_override(run / "screening", "P1", "include", "relevant, with a comma", "")
    assert [r.paper_id for r in review_queue(run, "screening")] == ["P2", "P4"]
    assert load_overrides(run / "screening")["P1"] == ("include", "relevant, with a comma", "")


def test_interactive_review_records_decisions(tmp_path):
    run = _setup(tmp_path)
    answers = iter(["x", "e", "commentary only", "e2", "k", "q"])
    out = []
    counts = interactive_review(run, "screening", ask=lambda _: next(answers), out=out.append)
    assert counts["exclude"] == 1 and counts["kept"] == 1
    ov = load_overrides(run / "screening")
    assert ov["P1"] == ("exclude", "commentary only", "E2")
    assert ov["P2"][0] == "maybe" and "confirmed" in ov["P2"][1]
    assert "P4" not in ov  # quitté avant
    assert any("Réponse non reconnue" in o for o in out)


def test_screening_applies_human_overrides(tmp_path, monkeypatch):
    import agents.screening as screening
    from agents.schemas import PICO, Protocol

    run = _setup(tmp_path)
    (run / "records_deduplicated.json").write_text(json.dumps([
        {"id": f"P{i}", "title": f"Title P{i}", "abstract": "a", "year": 2020} for i in range(1, 5)]), encoding="utf-8")
    protocol = Protocol(title="t", research_question="q", pico=PICO(population="p", intervention="i", comparator="", outcome="o", study_design=""),
                        inclusion_criteria=["I"], exclusion_criteria=["E"])
    sig = screening.hashlib.sha256((json.dumps(protocol.model_dump(), sort_keys=True) + screening.SYSTEM_REVIEWER
                                    + screening.STANCE_A + screening.STANCE_B + screening.SYSTEM_ARBITER).encode()).hexdigest()[:12]
    (run / "screening" / "protocol_signature.txt").write_text(sig)
    monkeypatch.setattr(screening, "get_llm", lambda: None)
    append_override(run / "screening", "P1", "include", "relevant", "")
    s = screening.run_screening(run, protocol)
    assert s["human_decisions"] == 1 and s["passed_to_fulltext"] == 3  # P1 (humain) + P2 + P3
    passed = {d["id"] for d in json.loads((run / "screening" / "included_for_fulltext.json").read_text(encoding="utf-8"))}
    assert passed == {"P1", "P2", "P3"}
