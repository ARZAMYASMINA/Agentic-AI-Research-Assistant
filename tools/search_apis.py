"""Clients des API académiques : PubMed, Semantic Scholar, arXiv, OpenAlex.

Chaque fonction renvoie (articles récupérés, nombre total de résultats disponibles).
"""
from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

from agents.schemas import Protocol
from tools.paper import Paper

log = logging.getLogger(__name__)

TIMEOUT = 60
HEADERS = {"User-Agent": "agentic-sysreview/0.1 (academic project)"}


def _get(url: str, params: dict | None = None, headers: dict | None = None, retries: int = 4) -> requests.Response:
    """GET avec nouvelles tentatives sur erreurs temporaires (429, 5xx, réseau)."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers={**HEADERS, **(headers or {})}, timeout=TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code} sur {url}")
                wait = 3.0 * attempt
                log.warning("%s -> nouvelle tentative dans %.0fs (%d/%d)", last, wait, attempt, retries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            if attempt < retries:
                time.sleep(3.0 * attempt)
    raise RuntimeError(f"Échec de la requête vers {url} : {last}")


def _year(text: str | None) -> int | None:
    m = re.search(r"(19|20)\d{2}", text or "")
    return int(m.group(0)) if m else None


def _clean_doi(doi: str | None) -> str:
    d = (doi or "").strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", d)
    return d


# --------------------------------------------------------------------------- PubMed
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _ncbi_params() -> dict:
    p = {"tool": "agentic_sysreview"}
    if os.getenv("NCBI_API_KEY"):
        p["api_key"] = os.environ["NCBI_API_KEY"]
    if os.getenv("NCBI_EMAIL"):
        p["email"] = os.environ["NCBI_EMAIL"]
    return p


def _parse_pubmed(xml_bytes: bytes) -> list[Paper]:
    root = ET.fromstring(xml_bytes)
    out: list[Paper] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
        t = art.find(".//ArticleTitle")
        title = "".join(t.itertext()).strip() if t is not None else ""
        if not title:
            continue
        parts = []
        for a in art.findall(".//Abstract/AbstractText"):
            text = "".join(a.itertext()).strip()
            label = a.get("Label")
            parts.append(f"{label}: {text}" if label and text else text)
        pub = art.find(".//Article/Journal/JournalIssue/PubDate")
        year = None
        if pub is not None:
            year = _year(pub.findtext("Year")) or _year(pub.findtext("MedlineDate"))
        authors = []
        for au in art.findall(".//AuthorList/Author"):
            last, fore = au.findtext("LastName"), au.findtext("ForeName")
            name = " ".join(x for x in (fore, last) if x) or (au.findtext("CollectiveName") or "")
            if name:
                authors.append(name)
        doi = ""
        for aid in art.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = _clean_doi(aid.text)
        out.append(
            Paper(
                title=title,
                abstract=" ".join(p for p in parts if p),
                authors=authors,
                year=year,
                journal=art.findtext(".//Article/Journal/Title") or "",
                doi=doi,
                pmid=pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                sources=["pubmed"],
                source_ids={"pubmed": pmid},
            )
        )
    return out


def search_pubmed(query: str, protocol: Protocol, max_results: int) -> tuple[list[Paper], int]:
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance", **_ncbi_params()}
    res = _get(f"{EUTILS}/esearch.fcgi", params).json()["esearchresult"]
    ids = res.get("idlist", [])
    total = int(res.get("count", 0))
    papers: list[Paper] = []
    for i in range(0, len(ids), 200):
        batch = ids[i : i + 200]
        r = _get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", **_ncbi_params()})
        papers += _parse_pubmed(r.content)
        time.sleep(0.4)  # limite NCBI : 3 requêtes/s sans clé
    return papers, total


# --------------------------------------------------------------------------- Semantic Scholar
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "title,abstract,year,authors,externalIds,venue,url"


def search_semantic_scholar(query: str, protocol: Protocol, max_results: int) -> tuple[list[Paper], int]:
    headers = {"x-api-key": os.environ["SEMANTIC_SCHOLAR_API_KEY"]} if os.getenv("SEMANTIC_SCHOLAR_API_KEY") else {}
    params: dict = {"query": query, "fields": S2_FIELDS}
    if protocol.date_from or protocol.date_to:
        params["year"] = f"{protocol.date_from}-{protocol.date_to}"
    papers: list[Paper] = []
    total = 0
    while len(papers) < max_results:
        data = _get(S2_URL, params, headers).json()
        total = int(data.get("total", 0))
        for d in data.get("data", []):
            ext = d.get("externalIds") or {}
            if not d.get("title"):
                continue
            papers.append(
                Paper(
                    title=d["title"].strip(),
                    abstract=(d.get("abstract") or "").strip(),
                    authors=[a.get("name", "") for a in d.get("authors") or [] if a.get("name")],
                    year=d.get("year"),
                    journal=d.get("venue") or "",
                    doi=_clean_doi(ext.get("DOI")),
                    pmid=str(ext.get("PubMed") or ""),
                    url=d.get("url") or "",
                    sources=["semantic_scholar"],
                    source_ids={"semantic_scholar": d.get("paperId", "")},
                )
            )
        token = data.get("token")
        if not token or not data.get("data"):
            break
        params["token"] = token
        time.sleep(1.0)
    return papers[:max_results], total


# --------------------------------------------------------------------------- arXiv
ARXIV_URL = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def search_arxiv(query: str, protocol: Protocol, max_results: int) -> tuple[list[Paper], int]:
    q = query
    if protocol.date_from or protocol.date_to:
        start = f"{protocol.date_from or '1991'}01010000"
        end = f"{protocol.date_to or '3000'}12312359"
        q = f"({query}) AND submittedDate:[{start} TO {end}]"
    papers: list[Paper] = []
    total, offset = 0, 0
    while len(papers) < max_results:
        page = min(100, max_results - len(papers))
        r = _get(ARXIV_URL, {"search_query": q, "start": offset, "max_results": page, "sortBy": "relevance", "sortOrder": "descending"})
        root = ET.fromstring(r.content)
        total = int(root.findtext("os:totalResults", default="0", namespaces=NS))
        entries = root.findall("atom:entry", NS)
        if not entries:
            break
        for e in entries:
            title = re.sub(r"\s+", " ", e.findtext("atom:title", default="", namespaces=NS)).strip()
            if not title:
                continue
            url = e.findtext("atom:id", default="", namespaces=NS)
            papers.append(
                Paper(
                    title=title,
                    abstract=re.sub(r"\s+", " ", e.findtext("atom:summary", default="", namespaces=NS)).strip(),
                    authors=[a.findtext("atom:name", default="", namespaces=NS) for a in e.findall("atom:author", NS)],
                    year=_year(e.findtext("atom:published", default="", namespaces=NS)),
                    journal="arXiv",
                    doi=_clean_doi(e.findtext("arxiv:doi", default="", namespaces=NS)),
                    url=url,
                    sources=["arxiv"],
                    source_ids={"arxiv": url.rsplit("/abs/", 1)[-1]},
                )
            )
        offset += len(entries)
        if offset >= total:
            break
        time.sleep(3.0)  # arXiv demande 3 s entre deux requêtes
    return papers, total


# --------------------------------------------------------------------------- OpenAlex
OPENALEX_URL = "https://api.openalex.org/works"


def _abstract_from_inverted(inv: dict | None) -> str:
    if not inv:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def search_openalex(query: str, protocol: Protocol, max_results: int) -> tuple[list[Paper], int]:
    filters = []
    if protocol.date_from:
        filters.append(f"from_publication_date:{protocol.date_from}-01-01")
    if protocol.date_to:
        filters.append(f"to_publication_date:{protocol.date_to}-12-31")
    params: dict = {
        "search": query,
        "per-page": min(200, max_results),
        "cursor": "*",
        "select": "id,doi,title,publication_year,authorships,abstract_inverted_index,primary_location,ids",
    }
    if filters:
        params["filter"] = ",".join(filters)
    if os.getenv("OPENALEX_EMAIL"):
        params["mailto"] = os.environ["OPENALEX_EMAIL"]
    papers: list[Paper] = []
    total = 0
    while len(papers) < max_results:
        data = _get(OPENALEX_URL, params).json()
        total = int(data.get("meta", {}).get("count", 0))
        results = data.get("results", [])
        for w in results:
            title = (w.get("title") or "").strip()
            if not title:
                continue
            loc = (w.get("primary_location") or {}).get("source") or {}
            pmid_url = (w.get("ids") or {}).get("pmid") or ""
            papers.append(
                Paper(
                    title=title,
                    abstract=_abstract_from_inverted(w.get("abstract_inverted_index")),
                    authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []],
                    year=w.get("publication_year"),
                    journal=loc.get("display_name") or "",
                    doi=_clean_doi(w.get("doi")),
                    pmid=pmid_url.rstrip("/").rsplit("/", 1)[-1] if pmid_url else "",
                    url=w.get("doi") or w.get("id") or "",
                    sources=["openalex"],
                    source_ids={"openalex": w.get("id", "")},
                )
            )
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor or not results:
            break
        params["cursor"] = next_cursor
        time.sleep(0.2)
    return papers[:max_results], total


SEARCHERS = {
    "pubmed": search_pubmed,
    "semantic_scholar": search_semantic_scholar,
    "arxiv": search_arxiv,
    "openalex": search_openalex,
}


def search(source: str, query: str, protocol: Protocol, max_results: int) -> tuple[list[Paper], int]:
    if source not in SEARCHERS:
        raise ValueError(f"Source inconnue : {source}")
    return SEARCHERS[source](query, protocol, max_results)
