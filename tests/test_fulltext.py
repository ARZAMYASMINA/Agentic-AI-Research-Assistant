from tools.fulltext import clean_pdf_text, jats_to_text

JATS = b"""<article><front><article-meta><title-group><article-title>AI in screening</article-title></title-group>
<abstract><p>Short abstract.</p></abstract></article-meta></front>
<body>
<sec><title>Methods</title><p>We included 100 women.</p>
  <sec><title>Reader study</title><p>Readers read twice.</p></sec>
</sec>
<sec><title>Results</title><p>Sensitivity was 90%.</p>
  <table-wrap><label>Table 1</label><caption><p>Performance</p></caption>
  <table><tbody><tr><td>AI</td><td>0.9</td></tr><tr><td>Human</td><td>0.8</td></tr></tbody></table></table-wrap>
</sec>
</body></article>"""


def test_jats_structure_and_no_duplication():
    text = jats_to_text(JATS)
    assert text.startswith("# AI in screening")
    assert "## Abstract" in text and "Short abstract." in text
    assert "## Methods" in text and "### Reader study" in text
    assert text.count("Readers read twice.") == 1  # les sections imbriquées ne sont pas dupliquées
    assert "We included 100 women." in text and "Sensitivity was 90%." in text


def test_jats_tables_are_kept_as_rows():
    text = jats_to_text(JATS)
    assert "[Table] Table 1 Performance" in text
    assert "AI | 0.9" in text and "Human | 0.8" in text


def test_jats_invalid_xml_returns_empty():
    assert jats_to_text(b"not xml at all") == ""


def test_clean_pdf_text():
    raw = "The sensi-\ntivity was  high.\n\n\n\nNext   paragraph."
    assert clean_pdf_text(raw) == "The sensitivity was high.\n\nNext paragraph."
