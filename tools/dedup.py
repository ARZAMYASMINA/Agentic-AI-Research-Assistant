"""Déduplication des articles issus de plusieurs bases (DOI -> PMID -> titre normalisé)."""
from __future__ import annotations

import re
import unicodedata

from tools.paper import Paper

MIN_TITLE_LEN = 15  # en dessous, la comparaison par titre est trop risquée


def norm_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _years_compatible(a: int | None, b: int | None) -> bool:
    return a is None or b is None or abs(a - b) <= 1


def _merge(base: Paper, other: Paper) -> None:
    if len(other.abstract) > len(base.abstract):
        base.abstract = other.abstract
    for f in ("journal", "doi", "pmid", "url"):
        if not getattr(base, f) and getattr(other, f):
            setattr(base, f, getattr(other, f))
    if base.year is None:
        base.year = other.year
    if not base.authors:
        base.authors = other.authors
    base.sources = sorted(set(base.sources) | set(other.sources))
    base.source_ids = {**other.source_ids, **base.source_ids}


def deduplicate(papers: list[Paper]) -> tuple[list[Paper], dict]:
    """Renvoie (articles uniques avec id P0001..., statistiques de déduplication)."""
    unique: list[Paper] = []
    by_doi: dict[str, int] = {}
    by_pmid: dict[str, int] = {}
    by_title: dict[str, list[int]] = {}
    reasons = {"doi": 0, "pmid": 0, "title": 0}

    def register(idx: int) -> None:
        p = unique[idx]
        if p.doi:
            by_doi.setdefault(p.doi, idx)
        if p.pmid:
            by_pmid.setdefault(p.pmid, idx)
        key = norm_title(p.title)
        if len(key) >= MIN_TITLE_LEN and idx not in by_title.setdefault(key, []):
            by_title[key].append(idx)

    for p in papers:
        match, reason = None, ""
        if p.doi and p.doi in by_doi:
            match, reason = by_doi[p.doi], "doi"
        elif p.pmid and p.pmid in by_pmid:
            match, reason = by_pmid[p.pmid], "pmid"
        else:
            key = norm_title(p.title)
            if len(key) >= MIN_TITLE_LEN:
                for idx in by_title.get(key, []):
                    if _years_compatible(unique[idx].year, p.year):
                        match, reason = idx, "title"
                        break
        if match is None:
            unique.append(p.model_copy(deep=True))
            register(len(unique) - 1)
        else:
            _merge(unique[match], p)
            register(match)
            reasons[reason] += 1

    for i, p in enumerate(unique, start=1):
        p.id = f"P{i:04d}"

    stats = {
        "records_in": len(papers),
        "records_unique": len(unique),
        "duplicates_removed": len(papers) - len(unique),
        "duplicates_by_doi": reasons["doi"],
        "duplicates_by_pmid": reasons["pmid"],
        "duplicates_by_title": reasons["title"],
    }
    return unique, stats
