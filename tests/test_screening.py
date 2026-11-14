import pytest

from agents.schemas import PICO, Protocol, ScreeningDecision
from agents.screening import combine, rule_check, summarize
from agents.schemas import ScreeningRecord
from tools.metrics import cohen_kappa
from tools.paper import Paper


def D(decision, crit="I1", conf="high"):
    return ScreeningDecision(rationale="r", criterion=crit, decision=decision, confidence=conf)


PROTOCOL = Protocol(
    title="t", research_question="q", pico=PICO(population="p", intervention="i", outcome="o"),
    inclusion_criteria=["a"], exclusion_criteria=["b"], date_from="2015", date_to="2024",
)


def test_kappa_perfect_and_partial():
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(0.5)
    assert cohen_kappa([], []) is None


def test_kappa_all_same_label():
    assert cohen_kappa([True, True], [True, True]) == 1.0


def test_combine_consensus_and_arbiter():
    assert combine(D("include"), D("include")) == "include"
    assert combine(D("exclude"), D("exclude")) == "exclude"
    assert combine(D("include"), D("maybe")) == "maybe"
    assert combine(D("exclude"), D("include")) is None
    assert combine(D("maybe"), D("exclude")) is None


def test_rule_check_date_range():
    old = Paper(id="P1", title="Old paper", year=2010)
    new = Paper(id="P2", title="Too recent", year=2026)
    ok = Paper(id="P3", title="In range", year=2020)
    unknown = Paper(id="P4", title="No year")
    assert rule_check(old, PROTOCOL).final_criterion == "RULE:date"
    assert rule_check(new, PROTOCOL).decided_by == "rule"
    assert rule_check(ok, PROTOCOL) is None
    assert rule_check(unknown, PROTOCOL) is None


def test_summarize_counts_and_kappa():
    def rec(pid, a, b, final, by="consensus"):
        return ScreeningRecord(
            paper_id=pid, title="t", reviewer_a=D(a) if a else None, reviewer_b=D(b) if b else None,
            final_decision=final, final_criterion="E1" if final == "exclude" else "none",
            final_rationale="x", decided_by=by,
        )

    records = [
        rec("P1", "include", "include", "include"),
        rec("P2", "exclude", "exclude", "exclude"),
        rec("P3", "include", "exclude", "exclude", by="arbiter"),
        rec("P4", None, None, "exclude", by="rule"),
    ]
    s = summarize(records)
    assert s["records_screened"] == 4
    assert s["excluded"] == 3 and s["included"] == 1
    assert s["passed_to_fulltext"] == 1
    assert s["arbiter_calls"] == 1 and s["excluded_by_rule"] == 1
    assert s["dual_screened"] == 3
    assert s["inter_reviewer_agreement"] == pytest.approx(0.667, abs=0.001)
