from agents.citations import MARKER, SearchQueries, SupportJudgement, claims, find_references, validate_support
from agents.schemas import PICO, Protocol
from tools.paper import Paper

PROTOCOL = Protocol(title="T", research_question="Q?", pico=PICO(population="p", intervention="i", outcome="o"),
                    inclusion_criteria=["a"], exclusion_criteria=["b"])
ABSTRACT = ("Breast cancer is the most commonly diagnosed cancer among women worldwide and a leading cause of cancer death. "
            "Screening mammography reduces breast cancer mortality but produces false positive results. " + "x " * 100)
PAPER = Paper(title="Global burden of breast cancer", abstract=ABSTRACT, authors=["Jane Doe"], year=2021,
              journal="Lancet", doi="10.1/abc")


def test_claims_take_the_sentence_before_each_marker():
    paras = [f"Intro text. Breast cancer is a leading cause of death {MARKER}. Mammography has limits {MARKER}.", "Objective."]
    assert claims(paras) == [(0, "Breast cancer is a leading cause of death"), (0, "Mammography has limits")]


def test_validate_support():
    claim = "Breast cancer is a leading cause of cancer death among women"
    good = SupportJudgement(rationale="r", candidate=1, support="supports",
                            quote="Breast cancer is the most commonly diagnosed cancer among women worldwide and a leading cause of cancer death.")
    assert validate_support(claim, good, [PAPER]) == ""
    invented = good.model_copy(update={"quote": "Breast cancer kills more women than any other disease in every country."})
    assert "not found" in validate_support(claim, invented, [PAPER])
    assert "partial" in validate_support(claim, good.model_copy(update={"support": "partial"}), [PAPER])
    assert "out of range" in validate_support(claim, good.model_copy(update={"candidate": 3}), [PAPER])
    off_topic = good.model_copy(update={"quote": "Screening mammography reduces breast cancer mortality but produces false positive results."})
    assert validate_support("Deep learning models are trained on large image datasets", off_topic, [PAPER])


def test_find_references_replaces_only_verified_markers(tmp_path, monkeypatch):
    import agents.citations as cit

    def fake_call(schema, system, user, agent, llm=None, retries=3):
        if schema is SearchQueries:
            return SearchQueries(queries=["breast cancer burden"])
        if "leading cause" in user.split("\n")[0]:
            return SupportJudgement(rationale="r", candidate=1, support="supports",
                                    quote="Breast cancer is the most commonly diagnosed cancer among women worldwide and a leading cause of cancer death.")
        return SupportJudgement(rationale="r", candidate=None, support="none", quote="")

    monkeypatch.setattr(cit, "structured_call", fake_call)
    paras = [f"Breast cancer is a leading cause of cancer death among women {MARKER}. "
             f"AI could replace radiologists {MARKER}.", f"Breast cancer is a leading cause of cancer death among women {MARKER}.",
             f"This review aims to compare AI with radiologists {MARKER}."]
    res = find_references(paras, PROTOCOL, tmp_path / "c.json", lambda: object(), search_fn=lambda q: [PAPER])
    assert res["paragraphs"][0].count("[R1]") == 1 and MARKER in res["paragraphs"][0]
    assert res["paragraphs"][1].endswith("[R1].")  # même source réutilisée, même numéro
    assert len(res["references"]) == 1 and len(res["references"][0]["quotes"]) == 2
    assert res["paragraphs"][2] == "This review aims to compare AI with radiologists."  # objectif : marqueur retiré
    assert any("rejected" in c["status"] for c in res["claims"])
    # cache
    again = find_references(paras, PROTOCOL, tmp_path / "c.json", lambda: 1 / 0, search_fn=lambda q: 1 / 0)
    assert again["paragraphs"] == res["paragraphs"]


def test_included_studies_are_not_used_as_background(tmp_path, monkeypatch):
    import agents.citations as cit

    monkeypatch.setattr(cit, "structured_call", lambda schema, *a, **k: SearchQueries(queries=["q"]) if schema is SearchQueries else None)
    res = find_references([f"Some claim about breast cancer screening {MARKER}."], PROTOCOL, tmp_path / "c.json",
                          lambda: object(), exclude_dois={"10.1/ABC"}, search_fn=lambda q: [PAPER])
    assert res["claims"][0]["status"] == "no candidate found" and MARKER in res["paragraphs"][0]


def test_topic_anchor_rejects_off_topic_source():
    from agents.citations import topic_anchors

    anchors = topic_anchors(["breast cancer screening", "screening mammography", "Breast Neoplasms", "Female", "Adult"])
    assert anchors == {"breast", "mammogr", "neoplas"}
    prostate = Paper(title="AI and radiomics in MRI-based prostate diagnostics",
                     abstract="Modern radiology is challenged to identify prostate cancer with high sensitivity and specificity. " * 4)
    j = SupportJudgement(rationale="r", candidate=1, support="supports",
                         quote="Modern radiology is challenged to identify prostate cancer with high sensitivity and specificity.")
    assert "off topic" in validate_support("Screening needs high sensitivity and specificity to limit overdiagnosis", j, [prostate], anchors)


def test_objective_paragraph_is_never_referenced():
    paras = [f"Background claim {MARKER}.", f"This review aims to assess AI {MARKER}."]
    assert claims(paras) == [(0, "Background claim")]


def test_word_overlap_uses_stems():
    from agents.citations import _words

    assert {"standa", "reader", "radiol"} <= _words("stand-alone readers in radiology") & _words("standalone reader by radiologists")
