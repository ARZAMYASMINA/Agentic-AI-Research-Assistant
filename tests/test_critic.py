import json

from agents.critic import QA, _codes_ok, check_protocol, check_staleness, check_trace
from agents.observability import trace_summary
from agents.schemas import PICO, Protocol

PROTOCOL = Protocol(
    title="t", research_question="q", pico=PICO(population="p", intervention="i", outcome="o"),
    inclusion_criteria=["a", "b"], exclusion_criteria=["c"], date_from="2015",
)


def test_codes_ok():
    assert _codes_ok("I1/E1", PROTOCOL) and _codes_ok("RULE:date", PROTOCOL) and _codes_ok("human", PROTOCOL)
    assert not _codes_ok("I3", PROTOCOL) and not _codes_ok("E2", PROTOCOL) and not _codes_ok("X9", PROTOCOL)


def test_protocol_checks(tmp_path):
    (tmp_path / "fulltext").mkdir()
    (tmp_path / "screening").mkdir()
    (tmp_path / "fulltext" / "included_for_extraction.json").write_text(json.dumps([
        {"id": "P1", "year": 2010, "eligibility": "include"}, {"id": "P2", "year": 2020, "eligibility": "maybe"}]))
    (tmp_path / "screening" / "decisions.jsonl").write_text(
        json.dumps({"final_decision": "exclude", "final_criterion": "E7"}) + "\n")
    qa = QA()
    check_protocol(tmp_path, PROTOCOL, qa)
    msgs = " ".join(f["message"] for f in qa.findings)
    assert "outside the protocol period: P1" in msgs and "still 'maybe'" in msgs and "E7" in msgs
    assert [f["level"] for f in qa.findings] == ["error", "warning", "warning"]


def test_staleness(tmp_path):
    import os
    import time

    (tmp_path / "report").mkdir()
    (tmp_path / "evidence").mkdir()
    ms = tmp_path / "report" / "manuscript.md"
    ms.write_text("x")
    old = time.time() - 100
    os.utime(ms, (old, old))
    (tmp_path / "evidence" / "human_overrides.csv").write_text("id,kind,name,value,note\n")
    qa = QA()
    check_staleness(tmp_path, qa)
    assert any("manuscript.md is older" in f["message"] for f in qa.findings)


def test_trace_summary_counts_calls_errors_and_schema_echo(tmp_path):
    rows = [
        {"agent": "swim", "attempt": 1, "seconds": 2.5, "output": {}},
        {"agent": "swim", "attempt": 1, "error": "you returned the JSON schema itself instead of data"},
        {"agent": "swim", "attempt": 2, "seconds": 3.0, "output": {}},
        {"agent": "eligibility_reviewer", "attempt": 1, "seconds": 1.0, "output": {}},
    ]
    (tmp_path / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    t = trace_summary(tmp_path)
    assert t["agents"]["swim"] == {"calls": 2, "errors": 1, "retries": 1, "seconds": 5.5, "schema_echo": 1}
    assert t["agents"]["eligibility"]["calls"] == 1 and t["total_calls"] == 3
    qa = QA()
    check_trace({"llm": t}, qa)
    assert "1 returned the JSON schema" in qa.findings[0]["message"]
