from agents.eligibility import _apply_override, _load_overrides, _flag_exclusions, not_retrieved, reason_label, summarize
from agents.schemas import PICO, Protocol, ScreeningDecision, ScreeningRecord
from tools.paper import Paper

PROTOCOL = Protocol(
    title="t", research_question="q", pico=PICO(population="p", intervention="i", outcome="o"),
    inclusion_criteria=["Primary study comparing AI with radiologists", "Screening population"],
    exclusion_criteria=["Narrative reviews and commentaries", "Animal studies"],
)


def D(decision, conf="high"):
    return ScreeningDecision(rationale="r", criterion="I1", decision=decision, confidence=conf)


def rec(pid, final, level="full_text", crit="none", a=None, b=None, by="consensus", review=False):
    return ScreeningRecord(
        paper_id=pid, title="t", stage="full_text", evidence_level=level, final_decision=final,
        final_criterion=crit, final_rationale="x", decided_by=by, needs_human_review=review,
        reviewer_a=D(a) if a else None, reviewer_b=D(b) if b else None,
    )


def test_reason_label_maps_codes_to_criteria_text():
    assert reason_label("I1", PROTOCOL).startswith("I1 not met: Primary study")
    assert reason_label("E1/I2", PROTOCOL).startswith("E1: Narrative reviews")
    assert "not retrieved" in reason_label("RULE:not_retrieved", PROTOCOL)
    assert reason_label("E9", PROTOCOL) == "E9"  # code inconnu : renvoyé tel quel


def test_not_retrieved_record():
    r = not_retrieved(Paper(id="P0132", title="McKinney"))
    assert r.evidence_level == "none" and r.needs_human_review
    assert "fulltext/pdf/P0132.pdf" in r.final_rationale


def test_summarize_prisma_numbers():
    records = [
        rec("P1", "include", a="include", b="include"),
        rec("P2", "exclude", crit="E1", a="exclude", b="exclude"),
        rec("P3", "exclude", crit="I1", a="include", b="exclude", by="arbiter", review=True),
        rec("P4", "maybe", level="abstract_only", a="maybe", b="maybe", review=True),
        not_retrieved(Paper(id="P5", title="x")),
    ]
    s = summarize(records, PROTOCOL)
    assert s["reports_sought"] == 5
    assert s["reports_not_retrieved"] == 1
    assert s["reports_assessed"] == 4
    assert s["assessed_on_full_text"] == 3 and s["assessed_on_abstract_only"] == 1
    assert s["included"] == 1 and s["excluded"] == 2 and s["maybe_human_decision"] == 1
    assert s["arbiter_calls"] == 1
    assert s["needs_human_review"] == 3  # P3, P4 et le rapport non récupéré
    assert sum(s["exclusion_reasons"].values()) == 2


def test_apply_override_marks_human_and_keeps_audit_trail():
    r = rec("P7", "exclude", crit="I2", a="exclude", b="exclude", review=True)
    h = _apply_override(r, "include", "Screening population confirmed", "")
    assert h.decided_by == "human" and h.final_decision == "include" and not h.needs_human_review
    assert "automatic decision was: exclude" in h.final_rationale


def test_human_exclusion_gets_a_reason():
    h = _apply_override(rec("P8", "maybe"), "exclude", "Commentary on P0131", "")
    assert h.final_criterion == "human"
    assert reason_label(h.final_criterion, PROTOCOL) == "Excluded on human review"
    assert _apply_override(rec("P9", "maybe"), "exclude", "x", "E1").final_criterion == "E1"


def test_human_exclusion_keeps_not_retrieved_status():
    r = not_retrieved(Paper(id="P10", title="x"))
    assert _apply_override(r, "exclude", "irrelevant by title", "").final_criterion == "RULE:not_retrieved"


def test_load_overrides_ignores_invalid_rows(tmp_path):
    (tmp_path / "human_overrides.csv").write_text(
        "id,decision,note,criterion\nP1,Include,ok,\nP2,exclude,commentary,E1\nP3,bogus,x,\n", encoding="utf-8"
    )
    assert _load_overrides(tmp_path) == {"P1": ("include", "ok", ""), "P2": ("exclude", "commentary", "E1")}
    assert _load_overrides(tmp_path / "missing") == {}


def test_every_automatic_exclusion_is_flagged_for_review():
    assert _flag_exclusions(rec("P11", "exclude", a="exclude", b="exclude")).needs_human_review
    assert not _flag_exclusions(rec("P12", "include", a="include", b="include")).needs_human_review


def test_short_label_cuts_on_word_boundary():
    from agents.eligibility import _short
    s = _short("Primary study comparing the performance of an AI algorithm (alone or combined with radiologists) with standard reading")
    assert s.endswith("…") and len(s) <= 61 and not s[:-1].endswith(" ")
    assert "radiolo…" not in s
    assert _short("Short criterion.") == "Short criterion"
