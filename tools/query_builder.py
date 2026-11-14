"""Construction déterministe des requêtes Boolean par source à partir des blocs de concepts."""
from __future__ import annotations

from agents.schemas import ConceptBlock, Protocol, SourceQuery

MAX_KEYWORDS_PER_BLOCK = 8
SUPPORTED_SOURCES = ("pubmed", "semantic_scholar", "arxiv", "openalex")

# "Check tags" MeSH : présents dans presque tous les articles, ils ne filtrent rien
GENERIC_MESH = {
    "humans", "animals", "female", "male", "adult", "aged", "aged, 80 and over",
    "middle aged", "young adult", "adolescent", "child", "infant",
}


def _clean(term: str) -> str:
    return term.replace('"', "").replace("\n", " ").strip()


def _keywords(block: ConceptBlock) -> list[str]:
    seen, out = set(), []
    for k in block.keywords:
        k = _clean(k)
        if k and k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)
    return out[:MAX_KEYWORDS_PER_BLOCK]


def _mesh(block: ConceptBlock) -> list[str]:
    return [m for m in (_clean(m) for m in block.mesh_terms) if m and m.lower() not in GENERIC_MESH]


def build_pubmed(blocks: list[ConceptBlock], protocol: Protocol) -> str:
    groups = []
    for b in blocks:
        parts = [f'"{k}"[tiab]' for k in _keywords(b)]
        parts += [f'"{m}"[MeSH Terms]' for m in _mesh(b)]
        groups.append("(" + " OR ".join(parts) + ")")
    query = " AND ".join(groups)
    if protocol.date_from or protocol.date_to:
        start = protocol.date_from or "1900"
        end = protocol.date_to or "3000"
        query += f' AND ("{start}"[dp] : "{end}"[dp])'
    return query


def build_semantic_scholar(blocks: list[ConceptBlock], protocol: Protocol) -> str:
    groups = ["(" + " | ".join(f'"{k}"' for k in _keywords(b)) + ")" for b in blocks]
    return " + ".join(groups)


def build_arxiv(blocks: list[ConceptBlock], protocol: Protocol) -> str:
    groups = ["(" + " OR ".join(f'all:"{k}"' for k in _keywords(b)) + ")" for b in blocks]
    return " AND ".join(groups)


def build_openalex(blocks: list[ConceptBlock], protocol: Protocol) -> str:
    groups = ["(" + " OR ".join(f'"{k}"' for k in _keywords(b)) + ")" for b in blocks]
    return " AND ".join(groups)


_BUILDERS = {
    "pubmed": build_pubmed,
    "semantic_scholar": build_semantic_scholar,
    "arxiv": build_arxiv,
    "openalex": build_openalex,
}


def build_queries(
    blocks: list[ConceptBlock], protocol: Protocol, sources: list[str] | None = None
) -> list[SourceQuery]:
    sources = sources or list(SUPPORTED_SOURCES)
    unknown = [s for s in sources if s not in _BUILDERS]
    if unknown:
        raise ValueError(f"Sources non supportées : {unknown}")
    return [SourceQuery(source=s, query=_BUILDERS[s](blocks, protocol)) for s in sources]
