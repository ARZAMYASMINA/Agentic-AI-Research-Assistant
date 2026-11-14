from agents.manuscript import check_paragraph, split_sentences, study_block
from agents.schemas import DomainResult, EvidenceRecord, FieldValue
from tools.paper import Paper

BLOCKS = {
    "P0015": "[P0015] Hickman et al. (2023)\n  auc: DL-1: 0.87 (CI 0.85-0.89)\n  population: n=78,849 examinations",
    "P0165": "[P0165] Leibig et al. (2022)\n  sensitivity: Radiologist: internal 85.7%; AI standalone: internal 84.2%",
}


def test_paragraph_with_correct_citations_passes():
    text = ("Hickman et al. reported an AUC of 0.87 for DL-1 on 78,849 examinations [P0015]. "
            "In the German programme, AI reached 84.2% sensitivity versus 85.7% for radiologists [P0165].")
    assert check_paragraph(text, BLOCKS) == []


def test_number_attributed_to_wrong_study_is_flagged():
    text = "AI reached a sensitivity of 85.7% [P0015]."
    problems = check_paragraph(text, BLOCKS)
    assert problems and "85.7" in problems[0] and "P0015" in problems[0]


def test_computed_number_is_flagged():
    text = "The pooled sensitivity was 84.95% [P0165]."  # moyenne inventée
    assert any("84.95" in p for p in check_paragraph(text, BLOCKS))


def test_unknown_id_and_uncited_numbers_are_flagged():
    problems = check_paragraph("One study reported 0.87 AUC. Another found 0.99 [P0999].", BLOCKS)
    joined = " ".join(problems)
    assert "without citation" in joined and "P0999" in joined


def test_paragraph_without_any_citation_is_flagged():
    assert "no study cited" in check_paragraph("AI may help radiologists in screening.", BLOCKS)


def test_ids_are_not_counted_as_numbers():
    # « P0015 » ne doit pas être lu comme le nombre 0015
    assert check_paragraph("This retrospective study evaluated three algorithms [P0015].", BLOCKS) == []


def test_split_sentences_keeps_decimals_and_citations():
    s = split_sentences("AUC was 0.87 [P0015]. Sensitivity was 84.2% [P0165]. (See Table 3.)")
    assert len(s) == 3 and s[0].endswith("[P0015].")


def test_study_block_hides_unverified_values():
    rec = EvidenceRecord(
        paper_id="P0001", title="t", year=2024, evidence_level="full_text", overall_risk="high",
        extracted={
            "auc": FieldValue(value="0.91", found=True, evidence_verified=True, numbers_verified=True),
            "sensitivity": FieldValue(value="88.8%", found=True, evidence_verified=False, numbers_verified=False),
            "specificity": FieldValue(value="93.4%", found=True, human=True),
        },
        risk_of_bias=[DomainResult(domain="index_test", judgement="high", llm_judgement="high", rationale="r")],
    )
    block = study_block(rec, Paper(title="t", authors=["Jane Doe", "John Roe"]))
    assert "auc: 0.91" in block and "specificity: 93.4%" in block
    assert "88.8" not in block and "not available" in block
    assert "Doe et al." in block and "overall risk of bias (QUADAS-2): high" in block


def test_markdown_to_docx(tmp_path):
    from docx import Document

    from tools.docx_export import markdown_to_docx

    md = ("# Title\n\n**Bold** intro with *italic*.\n\n## Methods\n\n- item one\n\n"
          "| ID | AUC |\n|---|---|\n| P0015 | 0.87 |\n\n> ⚠ flagged\n\n```text\nquery\n```\n")
    path = markdown_to_docx(md, tmp_path / "m.docx")
    doc = Document(path)
    texts = [p.text for p in doc.paragraphs]
    assert "Title" in texts and "Methods" in texts and "item one" in texts
    assert doc.tables[0].cell(1, 1).text == "0.87"


FACTS = {
    "P0015": {"level": "full_text", "overall": "high", "design": "retrospective cohort"},
    "P0165": {"level": "full_text", "overall": "high", "design": "retrospective cohort"},
    "P0065": {"level": "full_text", "overall": "high", "design": "prospective cohort"},
    "P0013": {"level": "abstract_only", "overall": "unclear", "design": "retrospective multireader study"},
}


def test_segments_attach_each_clause_to_its_citation():
    from agents.manuscript import segments

    segs = segments("AUC was 0.87 [P0015] while sensitivity was 84.2% [P0165].")
    assert [ids for _, ids in segs] == [{"P0015"}, {"P0165"}]
    assert "0.87" in segs[0][0] and "84.2" in segs[1][0]
    merged = segments("Two studies agree [P0015][P0165].")
    assert merged[0][1] == {"P0015", "P0165"}


def test_number_checked_per_clause_not_per_sentence():
    blocks = {**BLOCKS, "P0065": "[P0065] x\n  other: 579,583 exams"}
    ok = "AUC was 0.87 [P0015], and 84.2% sensitivity [P0165]."
    swapped = "AUC was 84.2% [P0015], and 0.87 sensitivity [P0165]."
    assert check_paragraph(ok, blocks) == []
    assert len(check_paragraph(swapped, blocks)) == 2


def test_wrong_design_evidence_level_and_risk_are_flagged():
    blocks = {k: f"[{k}] data" for k in FACTS}
    p = check_paragraph("Studies used retrospective designs [P0015, P0065].", blocks, FACTS)
    assert any("retrospective" in x and "P0065" in x for x in p)
    p = check_paragraph("Shen et al., from abstract evidence only, found a gain [P0015].", blocks, FACTS)
    assert any("abstract only" in x for x in p)
    p = check_paragraph("Reports at high risk of bias [P0015, P0013] were common.", blocks, FACTS)
    assert any("P0013" in x and "unclear" in x for x in p)
    assert check_paragraph("Retrospective cohorts at high risk of bias [P0015, P0165].", blocks, FACTS) == []
    assert check_paragraph("A prospective cohort [P0065] and an abstract-only study [P0013].", blocks, FACTS) == []


def test_reference_initials_keep_name_particles():
    from agents.manuscript import _initials

    assert _initials("Cornelis Jan van Rooden") == "van Rooden CJ"
    assert _initials("Sarah E Hickman") == "Hickman SE"


def test_no_number_mode_flags_any_digit_but_not_ids():
    ok = "AI showed higher specificity than radiologists in three studies [P0015, P0165]."
    assert check_paragraph(ok, BLOCKS, allow_numbers=False) == []
    bad = check_paragraph("Sensitivity was 84.2% versus 85.7% [P0165].", BLOCKS, allow_numbers=False)
    assert len(bad) == 1 and "84.2" in bad[0] and "Table 3" in bad[0]


def test_introduction_checks():
    from agents.manuscript import check_introduction

    ok = ["Breast cancer is a leading cause of death [REFERENCE NEEDED].",
          "AI can read mammograms [REFERENCE NEEDED].",
          "Evidence is inconsistent [REFERENCE NEEDED].",
          "This review asks how AI compares with radiologists."]
    assert check_introduction(ok) == []
    bad = ["Breast cancer kills 685,000 women a year [REFERENCE NEEDED].",
           "AI was studied by McKinney et al. in Nature.",
           "Results from [P0015] were promising [REFERENCE NEEDED].",
           "Objective."]
    p = " ".join(check_introduction(bad))
    assert "paragraph 1: contains numbers (685,000)" in p
    assert "paragraph 2: names an author" in p and "paragraph 2: background paragraph without" in p
    assert "paragraph 3: cites an included study ID" in p


def test_docx_embeds_figure(tmp_path):
    from docx import Document
    from PIL import Image

    from tools.docx_export import markdown_to_docx

    Image.new("RGB", (40, 20), "white").save(tmp_path / "fig.png")
    doc = Document(markdown_to_docx("# T\n\n![Figure 1. Flow.](fig.png)\n\n![Missing.](nope.png)\n", tmp_path / "m.docx"))
    assert len(doc.inline_shapes) == 1
    texts = [p.text for p in doc.paragraphs]
    assert "Figure 1. Flow." in texts and any("image not found" in t for t in texts)
