"""Export BibTeX des études incluses."""
from __future__ import annotations

import re
import unicodedata

from tools.paper import Paper

_SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}
MAX_AUTHORS = 10


def _esc(s: str) -> str:
    return "".join(_SPECIAL.get(c, c) for c in s)


def _ascii(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode())


def bibtex_key(p: Paper) -> str:
    """Clé unique et stable : NomDuPremierAuteur + année + identifiant interne (ex. Lang2023P0015)."""
    surname = _ascii(p.authors[0].split()[-1]) if p.authors and p.authors[0].split() else ""
    return f"{surname or 'Anon'}{p.year or 'nd'}{p.id}"


def to_bibtex(papers: list[Paper]) -> str:
    entries = []
    for p in papers:
        kind = "article" if p.journal and p.journal != "arXiv" else "misc"
        authors = [a for a in p.authors if a][:MAX_AUTHORS]
        author_field = " and ".join(authors) + (" and others" if len([a for a in p.authors if a]) > MAX_AUTHORS else "")
        notes = []
        if p.pmid:
            notes.append(f"PMID: {p.pmid}")
        if p.journal == "arXiv":
            notes.append("arXiv preprint")
        fields = [
            ("title", "{" + _esc(p.title.rstrip(".")) + "}"),  # accolades internes : conserve la casse
            ("author", _esc(author_field)),
            ("journal", _esc(p.journal) if kind == "article" else ""),
            ("year", str(p.year) if p.year else ""),
            ("doi", p.doi),
            ("url", p.url),
            ("note", "; ".join(notes)),
        ]
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
        entries.append(f"@{kind}{{{bibtex_key(p)},\n{body}\n}}")
    return "\n\n".join(entries) + ("\n" if entries else "")
