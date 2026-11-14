import json

from agents.schemas import EvidenceRecord, FieldValue
from agents.swim import Comparison, StudyComparisons, direction, render, sign_test, study_direction, validate

# lignes réelles de P0043 (résumé), un cas où le modèle avait inversé les groupes
LINES = {
    "sensitivity": "AI not reported; radiologists without AI: 85.44% [95% CI: 83.22%, 87.65%]; with AI: 87.69% [95% CI: 85.63%, 89.75%]",
    "specificity": "AI: 89.64% [95% CI: 85.34%, 93.94%]; radiologists: 77.34% [95% CI: 75.82%, 78.87%]",
    "auc": "AI stand-alone: 0.93 [95% CI: 0.92, 0.94]; radiologists without AI: 0.90 [95% CI: 0.86, 0.93]; with AI: 0.92 [95% CI: 0.88, 0.96]",
}


def C(outcome, role, il, iv, cl, cv):
    return Comparison(outcome=outcome, ai_role=role, intervention_label=il, intervention_value=iv,
                      comparator_label=cl, comparator_value=cv)


def test_valid_comparisons_pass():
    assert validate(C("sensitivity", "assisted", "with AI", "87.69", "radiologists without AI", "85.44"), LINES) == ""
    assert validate(C("specificity", "standalone", "AI", "89.64%", "radiologists", "77.34%"), LINES) == ""
    assert validate(C("auc", "standalone", "AI stand-alone", "0.93", "radiologists without AI", "0.90"), LINES) == ""


def test_swapped_groups_are_rejected():
    # l'erreur réelle du LLM : 89.64 (IA seule) présenté comme lecture avec IA, 77.34 comme sans IA
    assert validate(C("specificity", "assisted", "with AI", "89.64", "without AI", "77.34"), LINES)
    # valeur du comparateur attribuée à l'IA
    assert validate(C("specificity", "standalone", "AI", "77.34", "radiologists", "89.64"), LINES)
    # « AI » ne désigne pas « radiologists without AI »
    assert validate(C("sensitivity", "standalone", "AI", "85.44", "radiologists", "87.69"), LINES)


def test_confidence_bound_and_wrong_line_are_rejected():
    assert validate(C("specificity", "standalone", "AI", "85.34", "radiologists", "77.34"), LINES)  # borne d'IC
    assert validate(C("auc", "standalone", "AI", "89.64", "radiologists", "77.34"), LINES)  # mauvaise ligne
    assert "no verified data line" in validate(C("reading_time", "assisted", "with AI", "1", "without AI", "2"), LINES)


def test_direction_rules():
    assert direction(C("sensitivity", "assisted", "a", "87.69", "b", "85.44")) == "▲"
    assert direction(C("specificity", "standalone", "a", "89.5", "b", "93.4")) == "▼"
    assert direction(C("sensitivity", "standalone", "a", "84.2", "b", "85.7")) == "◆"  # 1.75 % < 2 %
    assert direction(C("reading_time", "assisted", "a", "48.52", "b", "54.41")) == "▲"  # plus court = mieux
    assert direction(C("recall_rate", "assisted", "a", "11.1", "b", "10.6")) == "▼"
    assert direction(C("sensitivity", "standalone", "a", "84.2", "b", "85.7"), threshold=0.01) == "▼"


def test_sign_test_and_study_direction():
    assert sign_test(0, 0) is None
    assert sign_test(3, 0) == 0.25
    assert abs(sign_test(5, 1) - 0.21875) < 1e-9
    assert study_direction(["▲", "▲"]) == "▲" and study_direction(["▲", "◆"]) == "◐"


def _record(pid, lines, level="full_text", overall="high"):
    return EvidenceRecord(
        paper_id=pid, title="t", evidence_level=level, overall_risk=overall,
        extracted={k: FieldValue(value=v, found=True, human=True) for k, v in lines.items()},
    )


def test_extract_comparisons_retries_and_keeps_only_valid(tmp_path, monkeypatch):
    import agents.swim as swim

    calls = []

    def fake_call(schema, system, user, agent, llm=None, retries=3):
        calls.append(user)
        if len(calls) == 1:  # 1er essai : une bonne paire, une paire inversée
            return StudyComparisons(comparisons=[
                C("specificity", "standalone", "AI", "89.64", "radiologists", "77.34"),
                C("specificity", "assisted", "with AI", "89.64", "without AI", "77.34"),
            ])
        return StudyComparisons(comparisons=[C("sensitivity", "assisted", "with AI", "87.69", "radiologists without AI", "85.44")])

    monkeypatch.setattr(swim, "structured_call", fake_call)
    res = swim.extract_comparisons([_record("P0043", LINES)], tmp_path / "swim.json", lambda: object())
    r = res["P0043"]
    assert len(calls) == 2 and "failed automatic checks" in calls[1]
    assert {c["outcome"] for c in r["valid"]} == {"specificity", "sensitivity"}
    assert len(r["rejected"]) == 1
    # cache : pas de nouvel appel
    swim.extract_comparisons([_record("P0043", LINES)], tmp_path / "swim.json", lambda: object())
    assert len(calls) == 2
    assert json.loads((tmp_path / "swim.json").read_text(encoding="utf-8"))["P0043"]["valid"]


def test_render_counts_studies_and_writes_text():
    results = {
        "P0043": {"valid": [C("specificity", "standalone", "AI", "89.64", "radiologists", "77.34").model_dump()], "rejected": []},
        "P0165": {"valid": [C("specificity", "standalone", "AI standalone internal", "89.5", "Radiologist internal", "93.4").model_dump(),
                            C("specificity", "standalone", "AI standalone external", "91.3", "Radiologist external", "93.4").model_dump()],
                  "rejected": [{"reason": "x"}]},
        "P0015": {"valid": [], "rejected": []},
    }
    facts = {"P0043": {"level": "abstract_only", "overall": "unclear"}, "P0165": {"level": "full_text", "overall": "high"},
             "P0015": {"level": "full_text", "overall": "high"}}
    out = render(results, facts, 0.02)
    assert "| P0043 | standalone | ▲ |" in out["table"] and "| P0165 | standalone | ▼ |" in out["table"]
    assert "favoured the AI arm in 1 [P0043]" in out["synthesis"] and "favoured the comparator in 1 [P0165]" in out["synthesis"]
    assert out["n_valid"] == 3 and out["n_rejected"] == 1 and out["without"] == "P0015"
    assert "insufficient to draw a firm conclusion" in out["conclusion"]


def test_structured_call_rejects_echoed_schema(monkeypatch):
    import pytest

    from agents import llm as llm_mod

    class Resp:
        def __init__(self, c):
            self.content = c

    class FakeLLM:
        model = "fake"
        n = 0

        def model_copy(self, update=None):
            return self

        def invoke(self, messages):
            FakeLLM.n += 1
            return Resp(json.dumps(StudyComparisons.model_json_schema()))

    monkeypatch.setattr(llm_mod, "_trace", lambda rec: None)
    with pytest.raises(RuntimeError):
        llm_mod.structured_call(StudyComparisons, "s", "u", agent="t", llm=FakeLLM())
    assert FakeLLM.n == 3


def test_role_is_derived_from_labels():
    from agents.swim import arm, role_from_labels

    assert arm("AI") == "ai_alone" and arm("AI standalone internal") == "ai_alone" and arm("DL-1") == "ai_alone"
    assert arm("with AI") == "ai_assisted" and arm("AI-supported") == "ai_assisted"
    assert arm("radiologists without AI") == "comparator" and arm("unaided") == "comparator"
    assert arm("Radiologist internal") == "comparator" and arm("First reader") == "comparator"
    # l'erreur réelle : IA seule contre radiologues déclarée « assisted » par le modèle
    assert role_from_labels(C("specificity", "assisted", "AI", "89.64", "radiologists", "77.34")) == "standalone"
    assert role_from_labels(C("sensitivity", "standalone", "with AI", "87.69", "radiologists without AI", "85.44")) == "assisted"


def test_hints_detect_missed_comparisons():
    from agents.swim import hints

    lines = {"sensitivity": "Radiologist internal: 85.7% (95% CI 83.6-87.9); AI standalone internal: 84.2% (82.4-85.8)",
             "auc": "DL-1: 0.87 (CI 0.85-0.89); DL-2: 0.89 (CI 0.87-0.90)"}
    assert hints(lines) == ["sensitivity"]


def test_empty_answer_with_visible_comparison_triggers_retry(tmp_path, monkeypatch):
    import agents.swim as swim

    lines = {"sensitivity": "Radiologist internal: 85.7%; AI standalone internal: 84.2%"}
    calls = []

    def fake_call(schema, system, user, agent, llm=None, retries=3):
        calls.append(user)
        if len(calls) == 1:
            return StudyComparisons(comparisons=[])
        return StudyComparisons(comparisons=[C("sensitivity", "assisted", "AI standalone internal", "84.2", "Radiologist internal", "85.7")])

    monkeypatch.setattr(swim, "structured_call", fake_call)
    r = swim.extract_comparisons([_record("P0165", lines)], tmp_path / "s.json", lambda: object())["P0165"]
    assert len(calls) == 2 and "no comparison was returned" in calls[1]
    assert r["valid"][0]["ai_role"] == "standalone" and r["valid"][0]["model_role"] == "assisted"


def test_conclusion_states_exact_counts():
    results = {
        "P0131": {"valid": [C("sensitivity", "standalone", "AI", "81.9", "Radiologists (first)", "77.4").model_dump()], "rejected": []},
        "P0165": {"valid": [C("sensitivity", "standalone", "AI standalone internal", "84.2", "Radiologist internal", "85.7").model_dump()], "rejected": []},
    }
    facts = {"P0131": {"level": "abstract_only", "overall": "unclear"}, "P0165": {"level": "full_text", "overall": "high"}}
    concl = render(results, facts, 0.02)["conclusion"]
    assert "Sensitivity with AI alone (2 studies): 1 favoured AI, 1 showed no clear difference" in concl
    assert "mostly" not in concl


def test_design_class_ignores_double_reading():
    from agents.manuscript import _design_class

    assert _design_class("retrospective case-control study; double-reader population-based screening cohort") == "retrospective study"
    assert _design_class("retrospective multireader multicase study") == "reader study"
    assert _design_class("reader study") == "reader study"
