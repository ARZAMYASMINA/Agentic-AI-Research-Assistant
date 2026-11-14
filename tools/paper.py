"""Format commun d'un article, quelle que soit la source."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Paper(BaseModel):
    id: str = ""  # attribué après déduplication : P0001, P0002, ...
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    url: str = ""
    sources: list[str] = Field(default_factory=list)  # bases où l'article a été trouvé
    source_ids: dict[str, str] = Field(default_factory=dict)  # identifiant dans chaque base
