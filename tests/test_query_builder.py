from agents.schemas import ConceptBlock, PICO, Protocol
from agents.search_strategy import make_queries
from tools.query_builder import build_queries

PROTOCOL = Protocol(
    title="t",
    research_question="q",
    pico=PICO(population="p", intervention="i", outcome="o"),
    inclusion_criteria=["a"],
    exclusion_criteria=["b"],
    date_from="2015",
)

BLOCKS = [
    ConceptBlock(concept="population", keywords=["breast cancer", "mammary neoplasm", "Breast Cancer"], mesh_terms=["Breast Neoplasms"]),
    ConceptBlock(concept="intervention", keywords=['deep "learning"', "AI"]),
]


def test_pubmed_query():
    q = {s.source: s.query for s in build_queries(BLOCKS, PROTOCOL, ["pubmed"])}["pubmed"]
    assert '"breast cancer"[tiab]' in q
    assert '"Breast Neoplasms"[MeSH Terms]' in q
    assert '"deep learning"[tiab]' in q  # guillemets internes nettoyés
    assert ") AND (" in q
    assert '"2015"[dp] : "3000"[dp]' in q


def test_dedup_keywords_case_insensitive():
    q = build_queries(BLOCKS, PROTOCOL, ["openalex"])[0].query
    assert q.count('"breast cancer"') + q.count('"Breast Cancer"') == 1


def test_other_sources():
    qs = {s.source: s.query for s in build_queries(BLOCKS, PROTOCOL)}
    assert " + " in qs["semantic_scholar"]
    assert 'all:"AI"' in qs["arxiv"]
    assert set(qs) == {"pubmed", "semantic_scholar", "arxiv", "openalex"}


def test_generic_mesh_removed():
    blocks = [
        ConceptBlock(
            concept="population",
            keywords=["breast cancer screening"],
            mesh_terms=["Breast Neoplasms", "Female", "Adult", "Aged"],
        )
    ]
    q = build_queries(blocks, PROTOCOL, ["pubmed"])[0].query
    assert "Breast Neoplasms" in q
    assert "Female" not in q and "Aged" not in q and "Adult" not in q


def test_outcome_block_excluded_even_with_long_concept_name():
    blocks = [
        ConceptBlock(concept="Population: Women undergoing screening", keywords=["breast cancer screening"]),
        ConceptBlock(concept="Intervention: AI as adjunct to radiologists", keywords=["deep learning"]),
        ConceptBlock(concept="Outcome: Diagnostic accuracy metrics", keywords=["diagnostic accuracy"]),
    ]
    normalized, queries = make_queries(blocks, PROTOCOL, ["pubmed"])
    assert [b.concept for b in normalized] == ["population", "intervention", "outcome"]
    q = queries[0].query
    assert "deep learning" in q and "breast cancer screening" in q
    assert "diagnostic accuracy" not in q
