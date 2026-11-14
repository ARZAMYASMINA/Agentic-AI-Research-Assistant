import json
import xml.etree.ElementTree as ET

from agents.prisma import collect_flow, render_svg
from agents.report import characteristics_md, results_md, rob_md
from agents.schemas import DomainResult, EvidenceRecord, FieldValue
from tools.bibtex import bibtex_key, to_bibtex
from tools.paper import Paper


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_run(tmp_path, passed=10, included=None):
    _write(tmp_path / "prisma_identification.json", {
        "identification": {
            "per_source": {"pubmed": {"hits_total": 100, "retrieved": 50}, "arxiv": {"hits_total": 5, "retrieved": 5}},
            "total_retrieved": 55, "duplicates_removed": 5, "records_unique": 50,
        },
        "errors": {},
    })
    _write(tmp_path / "screening" / "prisma_screening.json", {"records_screened": 50, "excluded": 40, "passed_to_fulltext": passed})
    _write(tmp_path / "fulltext" / "prisma_eligibility.json", {
        "reports_sought": 10, "reports_not_retrieved": 3, "reports_assessed": 7, "excluded": 2,
        "exclusion_reasons": {"Inclusion criterion not met - Primary study comparing AI with radiologists": 2},
    })
    included = included if included is not None else (
        [{"id": f"P{i}", "eligibility": "include", "evidence_level": "full_text"} for i in range(3)]
        + [{"id": f"P{i}", "eligibility": "maybe", "evidence_level": "abstract_only"} for i in range(3, 5)]
    )
    _write(tmp_path / "fulltext" / "included_for_extraction.json", included)
    return tmp_path


# --------------------------------------------------------------------------- PRISMA
def test_collect_flow_numbers_and_no_warning(tmp_path):
    flow = collect_flow(make_run(tmp_path))
    assert flow["identification"]["total_retrieved"] == 55 and flow["identification"]["duplicates_removed"] == 5
    assert flow["eligibility"]["assessed"] == 7
    assert flow["included"] == {"studies": 5, "include": 3, "provisional": 2, "full_text": 3, "abstract_only": 2}
    assert flow["warnings"] == []


def test_collect_flow_warns_on_inconsistent_counts(tmp_path):
    flow = collect_flow(make_run(tmp_path, passed=12, included=[{"id": "P1", "eligibility": "include", "evidence_level": "full_text"}]))
    assert len(flow["warnings"]) == 2  # passés au texte intégral != recherchés ; évalués - exclus != inclus


def test_render_svg_is_valid_xml_and_contains_the_numbers(tmp_path):
    svg = render_svg(collect_flow(make_run(tmp_path)))
    root = ET.fromstring(svg)  # lève une erreur si le XML est invalide
    assert root.tag.endswith("svg")
    text = " ".join(t.text or "" for t in root.iter() if t.tag.endswith("text"))
    for expected in ("n = 55", "n = 50", "n = 10", "n = 7", "n = 5"):
        assert expected in text
    assert "Duplicate records (n = 5)" in text and "pending human decision: 2" in text


def test_render_svg_escapes_special_characters(tmp_path):
    flow = collect_flow(make_run(tmp_path))
    flow["eligibility"]["reasons"] = {"Reason with <tags> & ampersand": 2}
    ET.fromstring(render_svg(flow))  # ne doit pas lever d'erreur


# --------------------------------------------------------------------------- BibTeX
def test_bibtex_key_and_entry():
    p = Paper(id="P0015", title="Triage using deep learning: a UK study.", authors=["Marie Lång", "John Smith"],
              year=2023, journal="Radiology", doi="10.1/abc", pmid="123")
    assert bibtex_key(p) == "Lang2023P0015"  # l'accent est décomposé puis retiré : clé ASCII sûre
    bib = to_bibtex([p])
    assert bib.startswith("@article{Lang2023P0015,")
    assert "title = {{Triage using deep learning: a UK study}}" in bib
    assert "author = {Marie Lång and John Smith}" in bib
    assert "doi = {10.1/abc}" in bib and "PMID: 123" in bib


def test_bibtex_escapes_and_arxiv_is_misc():
    p = Paper(id="P9", title="AI & screening: 50% faster_reads", authors=[], year=None, journal="arXiv")
    bib = to_bibtex([p])
    assert bib.startswith("@misc{AnonndP9,")
    assert r"AI \& screening: 50\% faster\_reads" in bib
    assert "journal" not in bib and "arXiv preprint" in bib


def test_bibtex_truncates_long_author_lists():
    p = Paper(id="P1", title="t", authors=[f"A{i} B{i}" for i in range(15)], journal="J")
    assert to_bibtex([p]).count(" and ") == 10 and "and others" in to_bibtex([p])


# --------------------------------------------------------------------------- tableaux
def rec(pid="P1", found=True, verified=True, nums=None, value="84.2% (95% CI 80-88) | pipe"):
    fv = FieldValue(value=value, evidence="e", found=found, evidence_verified=verified, numbers_verified=nums)
    return EvidenceRecord(
        paper_id=pid, title="t", year=2023, evidence_level="full_text", extracted={"sensitivity": fv},
        risk_of_bias=[DomainResult(domain="index_test", judgement="high", llm_judgement="high", rationale="r")],
        overall_risk="high",
    )


def test_tables_escape_pipes_and_flag_ungrounded_values():
    md = results_md([rec(verified=False, nums=False)])
    assert "⚠" in md  # valeur non étayée signalée
    assert "84.2% (95% CI 80-88) / pipe" in md  # le « | » ne casse pas le tableau
    assert "⚠" not in results_md([rec(verified=True)])


def test_missing_fields_and_rob_table():
    md = characteristics_md([rec()])
    assert "not reported" in md  # champs absents de l'extraction
    rob = rob_md([rec()])
    assert "| P1 | high | high |" in rob and "index test" in rob


def test_render_png_writes_a_readable_image(tmp_path):
    from PIL import Image

    from agents.prisma import render_png

    path = render_png(collect_flow(make_run(tmp_path)), tmp_path / "flow.png")
    with Image.open(path) as img:
        assert img.format == "PNG" and img.width > 1500 and img.height > 800
