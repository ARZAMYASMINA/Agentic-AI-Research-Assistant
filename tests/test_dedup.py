from tools.dedup import deduplicate, norm_title
from tools.paper import Paper


def P(title, doi="", pmid="", year=2020, abstract="", source="pubmed"):
    return Paper(title=title, doi=doi, pmid=pmid, year=year, abstract=abstract, sources=[source], source_ids={source: "x"})


def test_norm_title():
    assert norm_title("Deep-Learning: for Mammography!") == "deep learning for mammography"


def test_dedup_by_doi_and_merge():
    a = P("AI for breast cancer screening in mammography", doi="10.1/abc", abstract="short", source="pubmed")
    b = P("Totally different title here for the same doi", doi="10.1/abc", abstract="a much longer abstract", source="openalex")
    unique, stats = deduplicate([a, b])
    assert len(unique) == 1
    assert stats["duplicates_by_doi"] == 1
    assert unique[0].abstract == "a much longer abstract"
    assert unique[0].sources == ["openalex", "pubmed"]
    assert unique[0].id == "P0001"


def test_dedup_by_pmid():
    a = P("First title of a paper about screening", pmid="123")
    b = P("Another title with the same pubmed id", pmid="123", source="semantic_scholar")
    unique, stats = deduplicate([a, b])
    assert len(unique) == 1 and stats["duplicates_by_pmid"] == 1


def test_dedup_by_title_requires_compatible_year():
    a = P("Artificial intelligence in screening mammography", year=2020)
    b = P("Artificial Intelligence in Screening Mammography.", year=2021, source="arxiv")
    c = P("Artificial intelligence in screening mammography", year=2010, source="openalex")
    unique, stats = deduplicate([a, b, c])
    assert len(unique) == 2  # c est trop éloigné en année
    assert stats["duplicates_by_title"] == 1


def test_short_titles_not_merged():
    unique, _ = deduplicate([P("Editorial"), P("Editorial", source="openalex")])
    assert len(unique) == 2


def test_ids_sequential():
    unique, _ = deduplicate([P("Paper number one about topic A"), P("Paper number two about topic B")])
    assert [p.id for p in unique] == ["P0001", "P0002"]
