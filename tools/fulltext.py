"""Récupération des textes intégraux en open access.

Ordre d'essai : Europe PMC (XML JATS structuré, le plus propre) -> PDF open access trouvés par OpenAlex
-> PDF arXiv. Aucun contournement de paywall : seules les versions légalement ouvertes sont utilisées.
"""
from __future__ import annotations

import io
import logging
import os
import re
import xml.etree.ElementTree as ET

import requests

from tools.paper import Paper
from tools.search_apis import HEADERS

log = logging.getLogger(__name__)

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MIN_CHARS = 4000  # en dessous, le texte extrait est jugé inexploitable (PDF scanné, page de garde...)
MAX_PDF_BYTES = 30_000_000
MAX_PDF_PAGES = 40


def _fetch(url: str, params: dict | None = None, timeout: int = 45) -> requests.Response | None:
    """GET sans nouvelles tentatives : un 404 est ici une réponse normale (pas d'accès libre)."""
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    except requests.RequestException as e:
        log.debug("GET %s : %s", url, e)
        return None
    return r if r.status_code == 200 else None


def _download_pdf(url: str) -> bytes | None:
    try:
        with requests.get(url, headers=HEADERS, timeout=60, stream=True) as r:
            if r.status_code != 200:
                return None
            buf, size = io.BytesIO(), 0
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > MAX_PDF_BYTES:
                    return None
                buf.write(chunk)
    except requests.RequestException:
        return None
    data = buf.getvalue()
    return data if data[:5] == b"%PDF-" else None  # écarte les pages HTML d'accès / de connexion


# --------------------------------------------------------------------------- parsing
def _txt(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _table_text(tw: ET.Element) -> str:
    label = tw.findtext("label") or ""
    cap = tw.find("caption")
    head = f"[Table] {label} {_txt(cap) if cap is not None else ''}".strip()
    rows = []
    for tr in tw.iter("tr"):
        cells = [_txt(c) for c in tr if c.tag in ("td", "th")]
        if any(cells):
            rows.append(" | ".join(cells))
    return head + ("\n" + "\n".join(rows) if rows else "")


def jats_to_text(xml_bytes: bytes) -> str:
    """Texte brut structuré (titres Markdown, tableaux en lignes) depuis un XML JATS d'Europe PMC."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    parts: list[str] = []

    title = root.find(".//article-meta//article-title")
    if title is not None:
        parts.append("# " + _txt(title))
    abstract = root.find(".//article-meta/abstract")
    if abstract is not None:
        parts.append("## Abstract\n" + _txt(abstract))

    def walk(el: ET.Element, level: int) -> None:
        for child in el:
            if child.tag == "sec":
                t = child.find("title")
                if t is not None:
                    parts.append("#" * min(level + 1, 6) + " " + _txt(t))
                walk(child, level + 1)
            elif child.tag == "p":
                parts.append(_txt(child))
            elif child.tag == "table-wrap":
                parts.append(_table_text(child))
            elif child.tag == "fig":
                cap = child.find("caption")
                if cap is not None:
                    parts.append(f"[Figure] {_txt(cap)}")

    body = root.find(".//body")
    if body is not None:
        walk(body, 1)
    return "\n\n".join(p for p in parts if p).strip()


def clean_pdf_text(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # mots coupés en fin de ligne
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages[:MAX_PDF_PAGES]]
    except Exception as e:  # noqa: BLE001  PDF corrompu, chiffré, pypdf absent...
        log.warning("Lecture PDF impossible : %s", e)
        return ""
    return clean_pdf_text("\n\n".join(pages))


# --------------------------------------------------------------------------- sources
def europepmc_pmcid(paper: Paper) -> str | None:
    queries = []
    if paper.pmid:
        queries.append(f"EXT_ID:{paper.pmid} AND SRC:MED")
    if paper.doi:
        queries.append(f'DOI:"{paper.doi}"')
    for q in queries:
        r = _fetch(f"{EPMC}/search", {"query": q, "format": "json", "resultType": "lite", "pageSize": 1})
        if r is None:
            continue
        results = (r.json().get("resultList") or {}).get("result") or []
        if results and results[0].get("pmcid"):
            return results[0]["pmcid"]
    return None


def openalex_pdf_urls(paper: Paper) -> list[str]:
    if not paper.doi:
        return []
    params = {"select": "best_oa_location,open_access,locations"}
    if os.getenv("OPENALEX_EMAIL"):
        params["mailto"] = os.environ["OPENALEX_EMAIL"]
    r = _fetch(f"https://api.openalex.org/works/https://doi.org/{paper.doi}", params)
    if r is None:
        return []
    w = r.json()
    urls: list[str] = []

    def add(u: str | None) -> None:
        if u and u not in urls:
            urls.append(u)

    add((w.get("best_oa_location") or {}).get("pdf_url"))
    for loc in w.get("locations") or []:
        add(loc.get("pdf_url"))
    add((w.get("open_access") or {}).get("oa_url"))  # peut être une page HTML : _download_pdf la rejette
    return urls[:5]


def arxiv_pdf_urls(paper: Paper) -> list[str]:
    aid = paper.source_ids.get("arxiv")
    if aid:
        return [f"https://arxiv.org/pdf/{aid}"]
    m = re.search(r"arxiv\.org/abs/(\S+)", paper.url or "")
    return [f"https://arxiv.org/pdf/{m.group(1)}"] if m else []


def fetch_fulltext(paper: Paper) -> dict | None:
    """Renvoie {method, url, text, pdf} ou None si aucun texte intégral ouvert n'est trouvé."""
    pmcid = europepmc_pmcid(paper)
    if pmcid:
        url = f"{EPMC}/{pmcid}/fullTextXML"
        r = _fetch(url, timeout=60)
        if r is not None:
            text = jats_to_text(r.content)
            if len(text) >= MIN_CHARS:
                return {"method": "europepmc_xml", "url": url, "text": text, "pdf": None}

    for url in openalex_pdf_urls(paper) + arxiv_pdf_urls(paper):
        data = _download_pdf(url)
        if data:
            text = pdf_to_text(data)
            if len(text) >= MIN_CHARS:
                return {"method": "pdf", "url": url, "text": text, "pdf": data}
    return None
