"""Agent de références pour l'introduction : remplace les marqueurs [REFERENCE NEEDED] par des références vérifiées.

Pour chaque affirmation de contexte suivie d'un marqueur :
1. le LLM propose une ou deux requêtes de mots-clés ;
2. le code interroge PubMed et OpenAlex (articles avec résumé, toutes dates) ;
3. le LLM choisit le candidat dont le RÉSUMÉ soutient l'affirmation et en copie la phrase justificative ;
4. le code vérifie : la citation existe mot pour mot dans le titre/résumé du candidat choisi, elle partage des mots
   de contenu avec l'affirmation, et le modèle a répondu « supports » (pas « partial »).
Seule une référence vérifiée remplace le marqueur ([R1], [R2]...) ; sinon le marqueur reste pour l'équipe.
Tout est tracé dans report/intro_references.json (affirmation, requêtes, candidats, citation justificative).
Limite assumée : la vérification prouve que la phrase citée existe et touche au même sujet, pas que la source est la
meilleure possible ; l'équipe doit relire les références (annexe B du manuscrit).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agents.llm import structured_call
from agents.schemas import Protocol
from tools.paper import Paper
from tools.quotes import verify_quote

log = logging.getLogger(__name__)
MARKER = "[REFERENCE NEEDED]"
MAX_CANDIDATES = 6
MIN_SHARED = 3  # racines communes affirmation/citation (2 laissait passer « standard » ~ « standalone »)

_STOP = set("""a an the and or of in on for to with without by as at from is are was were be been being this that these
those it its their there which who whom whose than then such can may might could would should will not no more most less
also into over under between among within across about after before during while however both either neither each other
some any all many much very well often particularly especially including include includes use used using based""".split())


class SearchQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=2, description="1-2 short keyword queries (3-6 words each)")


class SupportJudgement(BaseModel):
    rationale: str = Field(description="1 sentence")
    candidate: int | None = Field(description="number of the candidate whose abstract supports the claim, or null")
    support: Literal["supports", "partial", "none"]
    quote: str = Field(description="one sentence copied word for word from that candidate's title or abstract; empty if none")


SYSTEM_QUERY = """You prepare literature searches to find a source for one background statement of a scientific
introduction. Return 1 or 2 short keyword queries (3 to 6 plain words each, no Boolean operators, no quotes) that would
retrieve a review article or a key study supporting the statement."""

SYSTEM_SUPPORT = """You check whether a candidate source supports a background statement of a scientific introduction.
Read the candidates (title and abstract). Choose the ONE candidate whose abstract directly supports the statement and
copy, word for word, the sentence of its title or abstract that supports it.
- "supports": the copied sentence states the same idea as the statement;
- "partial": related but not the same idea;
- "none": no candidate supports it (candidate = null, quote = "").
Never paraphrase the quote. Prefer review articles and general statements over narrow technical results."""


def _words(text: str) -> set[str]:
    """Racines (6 lettres, sans tiret) des mots de contenu : « stand-alone readers » ~ « standalone reader »."""
    return {w.replace("-", "")[:6] for w in re.findall(r"[a-z][a-z-]{3,}", text.lower()) if w not in _STOP}


_GENERIC = set("""screening cancer cancers digital full-field field full women woman over forty female male adult adults
aged patients patient population programs programme programmes study studies disease diseases computer-assisted computer
assisted human humans""".split())


def topic_anchors(population_terms: list[str]) -> set[str]:
    """Racines (7 lettres) des mots propres à la population de la revue (ex. breast, mammogr, neoplas).

    Une source de contexte doit en contenir au moins une : cela écarte les sources d'un autre domaine qui ne partagent
    que des mots génériques (sensitivity, overdiagnosis...)."""
    words = {w for t in population_terms for w in re.findall(r"[a-z][a-z-]{3,}", t.lower())}
    return {w[:7] for w in words if w not in _GENERIC and w not in _STOP}


def claims(paragraphs: list[str]) -> list[tuple[int, str]]:
    """(indice du paragraphe, affirmation) pour chaque marqueur : l'affirmation est la phrase qui le précède.

    Le dernier paragraphe (objectif de la revue) n'est jamais sourcé : un objectif ne se justifie pas par une référence."""
    out = []
    for k, p in enumerate(paragraphs[:-1] if len(paragraphs) > 1 else paragraphs):
        pos = 0
        for m in re.finditer(re.escape(MARKER), p):
            before = p[pos:m.start()].strip()
            sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z])", before)[-1] if before else ""
            sentence = sentence.replace(MARKER, "").strip(" .")
            if sentence:
                out.append((k, sentence))
            pos = m.end()
    return out


def _search(query: str, protocol: Protocol, n: int = 4) -> list[Paper]:
    from tools.search_apis import search_openalex, search_pubmed

    open_protocol = protocol.model_copy(update={"date_from": "", "date_to": ""})  # contexte : toutes dates
    found: list[Paper] = []
    for fn in (search_pubmed, search_openalex):
        try:
            papers, _ = fn(query, open_protocol, n)
            found += papers[:n]
        except Exception as e:  # noqa: BLE001  une source indisponible n'arrête pas la recherche
            log.warning("Recherche « %s » échouée (%s) : %s", query, fn.__name__, e)
    return found


def _dedupe(papers: list[Paper], exclude_dois: set[str]) -> list[Paper]:
    seen, out = set(), []
    for p in papers:
        key = (p.doi or "").lower() or re.sub(r"\W+", " ", p.title.lower()).strip()
        if not key or key in seen or (p.doi and p.doi.lower() in exclude_dois) or len(p.abstract or "") < 200:
            continue
        seen.add(key)
        out.append(p)
    return out[:MAX_CANDIDATES]


def validate_support(claim: str, j: SupportJudgement, candidates: list[Paper], anchors: set[str] | None = None) -> str:
    """Chaîne vide si la référence est acceptée, sinon la raison du rejet."""
    if j.support != "supports" or j.candidate is None:
        return f"model judgement: {j.support}"
    if not 1 <= j.candidate <= len(candidates):
        return "candidate number out of range"
    p = candidates[j.candidate - 1]
    if anchors and not any(a in f"{p.title} {p.abstract}".lower() for a in anchors):
        return f"source off topic (none of: {', '.join(sorted(anchors))})"
    if not verify_quote(j.quote, f"{p.title}\n{p.abstract}", min_words=6):
        return "supporting quote not found word for word in the abstract"
    shared = _words(claim) & _words(j.quote)
    if len(shared) < MIN_SHARED:
        return f"quote shares too few content words with the claim ({', '.join(sorted(shared)) or 'none'})"
    return ""


def find_references(paragraphs: list[str], protocol: Protocol, cache_file: Path, llm_factory,
                    exclude_dois: set[str] | None = None, search_fn=None, anchors: set[str] | None = None,
                    topic_hint: str = "") -> dict:
    """Résout les marqueurs ; renvoie {"paragraphs": [...] avec [R#], "references": [...], "claims": [...]}."""
    search_fn = search_fn or (lambda q: _search(q, protocol))
    exclude_dois = {d.lower() for d in (exclude_dois or set()) if d}
    sig = hashlib.sha256((SYSTEM_QUERY + SYSTEM_SUPPORT + json.dumps(paragraphs) + json.dumps(sorted(anchors or [])) + topic_hint + f"v6-{MIN_SHARED}").encode()).hexdigest()[:12]
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("signature") == sig:
            return cached
    llm = None
    refs: list[dict] = []
    log_claims: list[dict] = []
    resolved: dict[tuple[int, str], str] = {}
    for k, claim in claims(paragraphs):
        llm = llm or llm_factory()
        print(f"  [références] « {claim[:70]}... »", flush=True)
        entry: dict = {"paragraph": k + 1, "claim": claim}
        try:
            qs = structured_call(SearchQueries, SYSTEM_QUERY, f"Statement: {claim}", agent="citation_query", llm=llm).queries
            if topic_hint and anchors:  # requête sans le sujet de la revue -> on l'ajoute (sinon sources hors domaine)
                qs = [q if any(a in q.lower() for a in anchors) else f"{q} {topic_hint}" for q in qs]
            entry["queries"] = qs
            # sources déjà validées pour d'autres affirmations : candidates en tête (une revue couvre souvent plusieurs points)
            known = [Paper.model_validate(r["paper"]) for r in refs]
            cands = _dedupe(known + [p for q in qs for p in search_fn(q)], exclude_dois)
            entry["candidates"] = [f"{p.title} ({p.year}) {p.doi}" for p in cands]
            if not cands:
                entry["status"] = "no candidate found"
                log_claims.append(entry)
                continue
            listing = "\n\n".join(f"Candidate {i}: {p.title} ({p.year})\nAbstract: {p.abstract[:1500]}" for i, p in enumerate(cands, 1))
            user = f"Statement: {claim}\n\n{listing}"
            j = structured_call(SupportJudgement, SYSTEM_SUPPORT, user, agent="citation_support", llm=llm)
            why = validate_support(claim, j, cands, anchors)
            if why and j.support != "none":  # un nouvel essai, avec la raison du rejet
                entry["first_judgement"] = {**j.model_dump(), "rejected": why}
                j = structured_call(SupportJudgement, SYSTEM_SUPPORT,
                                    user + f"\n\nYour previous answer was rejected: {why}. Choose a candidate whose sentence states "
                                    "the same idea as the statement and copy that sentence exactly, or answer none.",
                                    agent="citation_support_retry", llm=llm)
                why = validate_support(claim, j, cands, anchors)
            entry["judgement"] = j.model_dump()
        except Exception as e:  # noqa: BLE001
            why = f"error: {str(e)[:150]}"
        if why:
            entry["status"] = f"rejected ({why})"
            log_claims.append(entry)
            continue
        p = cands[j.candidate - 1]
        key = (p.doi or p.title).lower()
        num = next((r["id"] for r in refs if r["key"] == key), None)
        if num is None:
            num = f"R{len(refs) + 1}"
            refs.append({"id": num, "key": key, "paper": p.model_dump(), "quotes": []})
        next(r for r in refs if r["id"] == num)["quotes"].append({"claim": claim, "quote": j.quote})
        resolved[(k, claim)] = num
        entry["status"] = f"accepted as {num}"
        log_claims.append(entry)

    # remplace chaque marqueur dont l'affirmation a été résolue, dans l'ordre
    new_paragraphs = []
    for k, p in enumerate(paragraphs):
        out, pos = [], 0
        for m in re.finditer(re.escape(MARKER), p):
            before = p[pos:m.start()]
            sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z])", before.strip())[-1].replace(MARKER, "").strip(" .") if before.strip() else ""
            out.append(before + (f"[{resolved[(k, sentence)]}]" if (k, sentence) in resolved else MARKER))
            pos = m.end()
        out.append(p[pos:])
        text = "".join(out)
        if len(paragraphs) > 1 and k == len(paragraphs) - 1:  # paragraphe d'objectif : marqueurs retirés
            text = re.sub(r"\s*" + re.escape(MARKER), "", text)
        new_paragraphs.append(text)
    result = {"signature": sig, "paragraphs": new_paragraphs, "references": refs, "claims": log_claims}
    cache_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
