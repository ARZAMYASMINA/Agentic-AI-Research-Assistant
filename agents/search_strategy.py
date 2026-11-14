"""Search Strategy Agent : concepts + synonymes + MeSH (LLM), requêtes Boolean (code)."""
from __future__ import annotations

from agents.llm import structured_call
from agents.schemas import ConceptBlock, ConceptBlocks, Protocol, SearchStrategy, SourceQuery
from tools.query_builder import build_queries

_CONCEPTS = ("population", "intervention", "outcome")

SYSTEM = """You are an information specialist (medical librarian) designing a systematic review search.
From the protocol, produce exactly three concept blocks.
The `concept` field must be EXACTLY one of: "population", "intervention", "outcome" (a single word, no description).
Rules for each block:
- 4 to 8 keywords: main term, synonyms, abbreviations, spelling variants, related phrases.
- Keywords are short phrases (1-4 words). No Boolean operators, no quotes, no field tags.
- NEVER use a generic word alone (such as "women", "patients", "adults", "accuracy", "sensitivity").
  Always make terms specific to the topic (e.g. "breast cancer screening", not "women").
- mesh_terms: for biomedical topics you MUST give 2 to 5 official, topic-specific MeSH headings
  (e.g. "Breast Neoplasms", "Mammography", "Artificial Intelligence", "Deep Learning").
  Never use generic check tags (Female, Male, Humans, Adult, Aged, Middle Aged).
  Use an empty list only for non-biomedical topics.
- Favor recall: include broader and narrower terms.
- The outcome block is kept for later extraction and is NOT used in the search query."""

USER = """Protocol:
Title: {title}
Question: {question}
Population: {population}
Intervention: {intervention}
Outcome: {outcome}

Produce the concept blocks."""


def normalize_blocks(blocks: list[ConceptBlock]) -> list[ConceptBlock]:
    """Ramène le nom de concept à population / intervention / outcome (le modèle ajoute parfois une description)."""
    out = []
    for b in blocks:
        c = b.concept.strip().lower()
        name = next((k for k in _CONCEPTS if c.startswith(k)), None) or next((k for k in _CONCEPTS if k in c), c)
        out.append(b.model_copy(update={"concept": name}))
    return out


def make_queries(
    blocks: list[ConceptBlock], protocol: Protocol, sources: list[str] | None = None
) -> tuple[list[ConceptBlock], list[SourceQuery]]:
    """Normalise les blocs puis construit les requêtes (population AND intervention uniquement)."""
    blocks = normalize_blocks(blocks)
    # Bonne pratique : les résultats (outcome) dans la requête réduisent le rappel
    query_blocks = [b for b in blocks if b.concept != "outcome"] or blocks
    return blocks, build_queries(query_blocks, protocol, sources)


def build_search_strategy(protocol: Protocol, sources: list[str] | None = None) -> SearchStrategy:
    blocks = structured_call(
        ConceptBlocks,
        system=SYSTEM,
        user=USER.format(
            title=protocol.title,
            question=protocol.research_question,
            population=protocol.pico.population,
            intervention=protocol.pico.intervention,
            outcome=protocol.pico.outcome,
        ),
        agent="search_strategy",
    ).blocks
    blocks, queries = make_queries(blocks, protocol, sources)
    return SearchStrategy(blocks=blocks, queries=queries)
