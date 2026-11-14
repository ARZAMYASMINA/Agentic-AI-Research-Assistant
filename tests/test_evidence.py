from agents.appraisal import TOOLS, abstract_only_results, finalize_judgements, overall_risk
from agents.evidence import summarize
from agents.extraction import DEFAULT_GROUPS, finalize_fields
from agents.schemas import DomainJudgement, DomainResult, EvidenceRecord, ExtractedField, FieldValue
from tools.quotes import numbers_in_source, verify_quote

SOURCE = (
    "We included 5,000 women aged 50-69 from the national screening programme.\n"
    "Sensitivity was 84.2% (95% CI 80-88) for the AI and 81.0% for\nradiologists; the AI’s AUC was 0.93."
)


# --------------------------------------------------------------------------- citations
def test_verify_quote_exact_and_normalized():
    assert verify_quote("Sensitivity was 84.2% (95% CI 80-88) for the AI", SOURCE)
    assert verify_quote("sensitivity was 84.2% (95% ci 80-88) for the ai.", SOURCE)  # casse + point final
    assert verify_quote("81.0% for radiologists", SOURCE)  # retour à la ligne dans la source
    assert verify_quote("the AI's AUC was 0.93", SOURCE)  # apostrophe typographique


def test_verify_quote_rejects_invented_or_too_short():
    assert not verify_quote("Sensitivity was 91.5% for the AI", SOURCE)
    assert not verify_quote("0.93", SOURCE)  # une seule « parole » : trop court pour prouver quoi que ce soit
    assert not verify_quote("", SOURCE)
    assert not verify_quote("quelque chose", "")


def test_verify_quote_with_ellipsis():
    assert verify_quote("We included 5,000 women ... national screening programme", SOURCE)
    assert not verify_quote("We included 5,000 women ... private clinics", SOURCE)


# --------------------------------------------------------------------------- extraction
def test_finalize_fields_verifies_and_fills_missing():
    specs = [f for g in DEFAULT_GROUPS for f in g["fields"]][:3]  # study_design, country_and_setting, population
    raw = [
        ExtractedField(name="population", evidence="We included 5,000 women aged 50-69", value="5,000 women aged 50-69", found=True),
        ExtractedField(name="study_design", evidence="a randomized trial of 9,000 women", value="RCT", found=True),  # citation inventée
    ]
    out = finalize_fields(raw, specs, SOURCE)
    assert set(out) == {s["name"] for s in specs}
    assert out["population"].found and out["population"].evidence_verified
    assert out["study_design"].found and not out["study_design"].evidence_verified
    assert out["country_and_setting"].value == "not reported" and not out["country_and_setting"].found


def test_finalize_fields_treats_not_reported_values_as_missing():
    specs = [{"name": "auc", "description": "d"}]
    raw = [ExtractedField(name="AUC", evidence="", value="Not reported", found=True)]
    assert finalize_fields(raw, specs, SOURCE)["auc"].found is False


# --------------------------------------------------------------------------- risque de biais
def J(domain, judgement, evidence=""):
    return DomainJudgement(domain=domain, evidence=evidence, rationale="r", judgement=judgement)


def test_unverified_quote_downgrades_to_unclear():
    names = list(TOOLS["QUADAS-2"])
    raw = [
        J("patient_selection", "low", "We included 5,000 women aged 50-69"),  # citation réelle
        J("index_test", "high", "the algorithm was tuned on the test set"),  # citation absente de la source
        J("reference_standard", "unclear"),
    ]
    res = {r.domain: r for r in finalize_judgements(raw, names, SOURCE)}
    assert res["patient_selection"].judgement == "low" and res["patient_selection"].evidence_verified
    assert res["index_test"].judgement == "unclear" and res["index_test"].llm_judgement == "high"
    assert "Downgraded" in res["index_test"].rationale
    assert res["flow_and_timing"].judgement == "unclear"  # domaine absent de la réponse du modèle


def test_abstract_only_is_always_unclear():
    res = abstract_only_results("QUADAS-2")
    assert len(res) == 4 and {r.judgement for r in res} == {"unclear"}


def test_overall_risk_rules():
    def R(*js):
        return [DomainResult(domain=f"d{i}", judgement=j, llm_judgement=j, rationale="") for i, j in enumerate(js)]

    assert overall_risk(R("low", "low")) == "low"
    assert overall_risk(R("low", "unclear")) == "unclear"
    assert overall_risk(R("low", "unclear", "high")) == "high"
    assert overall_risk([]) == "unclear"


# --------------------------------------------------------------------------- synthèse
def test_summarize_rates():
    def rec(pid, level, found, verified, risk):
        fv = {"a": FieldValue(value="x", evidence="e", found=found, evidence_verified=verified), "b": FieldValue(value="not reported")}
        return EvidenceRecord(paper_id=pid, title="t", evidence_level=level, extracted=fv, overall_risk=risk)

    s = summarize([rec("P1", "full_text", True, True, "low"), rec("P2", "abstract_only", True, False, "unclear")])
    assert s["studies"] == 2 and s["full_text"] == 1 and s["abstract_only"] == 1
    assert s["fields_requested"] == 4 and s["fields_found"] == 2
    assert s["quotes_verified"] == 1 and s["quote_verification_rate"] == 0.5
    assert s["overall_risk_of_bias"] == {"low": 1, "unclear": 1}
    assert s["field_found_rate"]["a"] == 1.0 and s["field_found_rate"]["b"] == 0.0
    assert s["grounded"] == 1 and s["grounded_rate"] == 0.5


# --------------------------------------------------------------------------- nombres
def test_numbers_in_source():
    src = "Specificity was 89.0% [801 of 900] and AUC 0.93; 78 849 patients, P = .46"
    assert numbers_in_source("89.0% [801 of 900]; P = .46", src) is True
    assert numbers_in_source("sensitivity 91.5%", src) is False  # nombre inventé ou calculé par le modèle
    assert numbers_in_source("78,849 women", src) is True  # séparateurs de milliers différents
    assert numbers_in_source("two readers, 95% CI", src) is None  # aucun nombre significatif


def test_grounded_property():
    assert FieldValue(value="x", found=True, evidence_verified=False, numbers_verified=True).grounded
    assert FieldValue(value="x", found=True, evidence_verified=True).grounded
    assert not FieldValue(value="x", found=True, evidence_verified=True, numbers_verified=False).grounded  # chiffre inventé
    assert not FieldValue(value="x", found=True, evidence_verified=False, numbers_verified=None).grounded
    assert not FieldValue(value="x", found=True, evidence_verified=False, numbers_verified=False).grounded
    assert not FieldValue(value="not reported").grounded


def test_finalize_fields_numbers_ground_table_values():
    specs = [{"name": "specificity", "description": "d"}, {"name": "auc", "description": "d"}]
    raw = [
        ExtractedField(name="specificity", evidence="Spec 89.0 [801/900]", value="89.0% (801 of 900)", found=True),  # citation reformatée
        ExtractedField(name="auc", evidence="", value="0.99", found=True),  # nombre absent de la source
    ]
    out = finalize_fields(raw, specs, "Specificity was 89.0% [801 of 900] and AUC 0.93")
    assert not out["specificity"].evidence_verified and out["specificity"].grounded
    assert out["auc"].numbers_verified is False and not out["auc"].grounded


def test_finalize_fields_merges_entries_split_by_reader():
    # Le modèle répond « sensitivity: AI », « sensitivity: Radiologists »... au lieu de « sensitivity » : rien ne doit être perdu
    specs = [{"name": "sensitivity", "description": "d"}, {"name": "auc", "description": "d"}]
    raw = [
        ExtractedField(name="sensitivity: AI", evidence="Sensitivity was 84.2% (95% CI 80-88) for the AI", value="84.2% (95% CI 80-88)", found=True),
        ExtractedField(name="sensitivity: Radiologists", evidence="81.0% for radiologists", value="81.0%", found=True),
        ExtractedField(name="sensitivity: combined", evidence="", value="not reported", found=False),
        ExtractedField(name="auc (AI)", evidence="the AI’s AUC was 0.93", value="0.93", found=True),
    ]
    out = finalize_fields(raw, specs, SOURCE)
    s = out["sensitivity"]
    assert s.found and s.evidence_verified and s.numbers_verified and s.grounded
    assert "AI: 84.2%" in s.value and "Radiologists: 81.0%" in s.value
    assert "combined" not in s.value  # l'entrée vide est ignorée
    assert out["auc"].value == "AI: 0.93" and out["auc"].evidence_verified


def test_merged_value_is_not_quote_verified_if_one_part_has_no_quote():
    specs = [{"name": "specificity", "description": "d"}]
    raw = [
        ExtractedField(name="specificity: AI", evidence="", value="84.2%", found=True),  # pas de citation pour ce morceau
        ExtractedField(name="specificity: Radiologists", evidence="81.0% for radiologists", value="81.0%", found=True),
    ]
    out = finalize_fields(raw, specs, SOURCE)["specificity"]
    assert not out.evidence_verified  # un morceau non cité empêche la validation par citation
    assert out.numbers_verified is True and out.grounded  # mais les nombres, eux, sont bien dans le texte


def test_middle_dot_decimals_lancet_style():
    src = "achieved a sensitivity of 84·2% (95% CI 82·4–85·8) and an AUROC of 0·982"
    assert numbers_in_source("AI: 84.2% (CI 82.4-85.8); AUC 0.982", src) is True
    assert numbers_in_source("AI: 84.3%", src) is False
    assert verify_quote("sensitivity of 84.2% (95% CI 82.4-85.8)", src)
    assert verify_quote("sensitivity of 84·2%", src)


def test_quote_in_rationale_is_recovered():
    src = "A double-blind RCT was not possible because reading radiologists could not use the device while blinded."
    raw = [DomainJudgement(domain="index_test", evidence="", judgement="high",
                           rationale="The text states 'reading radiologists could not use the device while blinded', so readers were unblinded.")]
    res = finalize_judgements(raw, ["index_test"], src)[0]
    assert res.judgement == "high" and res.evidence_verified
    assert res.evidence == "reading radiologists could not use the device while blinded"
    assert "recovered" in res.rationale


def test_invented_quote_in_rationale_still_downgrades():
    src = "Patients were enrolled at two sites."
    raw = [DomainJudgement(domain="patient_selection", evidence="", judgement="low",
                           rationale="The authors say 'all consecutive women were included without exclusion'.")]
    res = finalize_judgements(raw, ["patient_selection"], src)[0]
    assert res.judgement == "unclear" and not res.evidence_verified


def test_verify_quote_tolerates_pdf_split_words():
    src = "There were insufficient follow-up data to report interval cancers or cancer stage at diag nosis. Exams from an eli gible manufacturer."
    assert verify_quote("cancer stage at diagnosis", src)
    assert verify_quote("from an eligible manufacturer", src)
    assert not verify_quote("cancer grade at diagnosis", src)


def _rec_for_override():
    names = list(TOOLS["QUADAS-2"])
    rob = [DomainResult(domain=n, judgement="low", llm_judgement="low", rationale="auto") for n in names]
    return EvidenceRecord(
        paper_id="P1", title="t", evidence_level="full_text",
        extracted={"sensitivity": FieldValue(value="not reported"), "auc": FieldValue(value="0.98", found=True)},
        risk_of_bias=rob, overall_risk="low",
    )


def test_human_overrides_applied_on_top_without_changing_original():
    from agents.evidence import apply_evidence_overrides

    rec = _rec_for_override()
    overrides = [
        {"id": "P1", "kind": "field", "name": "sensitivity", "value": "Radiologists: 85.7%", "note": "checked in text"},
        {"id": "P1", "kind": "rob", "name": "flow_and_timing", "value": "high", "note": "authors admit partial verification"},
        {"id": "P1", "kind": "rob", "name": "no_such_domain", "value": "high", "note": ""},
        {"id": "P9", "kind": "field", "name": "auc", "value": "1", "note": ""},
    ]
    (out,), problems = apply_evidence_overrides([rec], overrides)
    assert out.extracted["sensitivity"].human and out.extracted["sensitivity"].grounded
    assert out.extracted["sensitivity"].value == "Radiologists: 85.7%"
    flow = next(d for d in out.risk_of_bias if d.domain == "flow_and_timing")
    assert flow.judgement == "high" and flow.human and flow.llm_judgement == "low"
    assert "automatic judgement was: low" in flow.rationale
    assert out.overall_risk == "high"
    assert len(problems) == 2
    # l'enregistrement automatique n'est pas modifié (audit)
    assert rec.overall_risk == "low" and rec.extracted["sensitivity"].value == "not reported"


def test_load_evidence_overrides_csv(tmp_path):
    from agents.evidence import load_evidence_overrides

    (tmp_path / "human_overrides.csv").write_text(
        "id,kind,name,value,note\nP1,rob,index_test,unclear,ok\nP1,rob,index_test,maybe,bad value\nP2,field,auc,0.944,ok\n",
        encoding="utf-8",
    )
    rows = load_evidence_overrides(tmp_path)
    assert [(r["id"], r["kind"], r["value"]) for r in rows] == [("P1", "rob", "unclear"), ("P2", "field", "0.944")]
