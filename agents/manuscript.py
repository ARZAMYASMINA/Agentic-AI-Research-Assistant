"""Synthesis & Writing Agent : brouillon de manuscrit PRISMA 2020 (report/manuscript.md + manuscript.docx).

Principe : tout ce qui peut être calculé l'est par le code, pas par le LLM.
- Méthodes, effectifs, limites : gabarits remplis avec les fichiers du run (aucune invention possible).
- Tableaux : ceux de l'étape 6a (agents/report.py), décisions humaines appliquées.
- Synthèse : méthode SWiM par direction des effets (agents/swim.py). Le LLM relève seulement les paires de
  valeurs comparées, étude par étude ; le code les vérifie, calcule les directions et rédige tout le texte.
  (La rédaction libre par le LLM, avec ou sans chiffres, a été abandonnée : erreurs d'étude, de groupe et de sens
  non détectables. Les fonctions de contrôle de paragraphes restent disponibles.)
Les réponses du LLM sont mises en cache (report/swim.json) : relancer sans changement ne rappelle pas le modèle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agents.evidence import current_records
from agents.extraction import DEFAULT_GROUPS
from agents.llm import get_llm, structured_call
from agents.prisma import SOURCE_NAMES, collect_flow
from agents.report import characteristics_md, results_md, rob_md
from agents.schemas import EvidenceRecord, Protocol
from tools.paper import Paper
from tools.quotes import significant_numbers

log = logging.getLogger(__name__)

DISCLAIMER = (
    "**AI-assisted draft — not for submission without human verification.** This manuscript was generated "
    "automatically by an agentic pipeline from the files of the review run. Methods and counts are filled from the "
    "run data. The abstract is written by the agent from the verified run data; the introduction is drafted by the "
    "language model from the protocol and every background statement is marked [REFERENCE NEEDED] for the team. "
    "The synthesis follows SWiM (direction of effect): a local language model only listed the compared "
    "values of each study; every comparison was verified, and all synthesis text was written, by code. Extraction "
    "values marked ⚠ are unverified and values marked † were corrected by a human. All content must be verified by "
    "the review team."
)


# --------------------------------------------------------------------------- données des études
_ID = re.compile(r"\bP\d{4}\b")
_SYNTH_FIELDS = [
    "study_design", "country_and_setting", "population", "number_of_cancers", "ai_system", "comparator",
    "reference_standard", "sensitivity", "specificity", "auc", "other_outcomes", "main_conclusion",
]


def _first_author(p: Paper | None) -> str:
    if not p or not p.authors:
        return "Anonymous"
    return p.authors[0].split()[-1] + (" et al." if len(p.authors) > 1 else "")


def study_block(rec: EvidenceRecord, paper: Paper | None) -> str:
    """Données d'une étude données au LLM : seules les valeurs étayées (ou corrigées par un humain) sont fournies."""
    rob = ", ".join(f"{d.domain.replace('_', ' ')} {d.judgement}" for d in rec.risk_of_bias)
    lines = [
        f"[{rec.paper_id}] {_first_author(paper)} ({rec.year or 'n.d.'})",
        f"  evidence: {'full text' if rec.evidence_level == 'full_text' else 'abstract only'}",
        f"  overall risk of bias (QUADAS-2): {rec.overall_risk} ({rob})",
    ]
    for name in _SYNTH_FIELDS:
        fv = rec.extracted.get(name)
        if fv is None or not fv.found:
            value = "not reported"
        elif not fv.grounded:
            value = "not available (value could not be verified in the text)"
        else:
            value = " ".join(fv.value.split())
        lines.append(f"  {name}: {value}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- contrôle des paragraphes
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text.strip()) if s.strip()]


def _numbers(text: str) -> set[str]:
    return significant_numbers(_ID.sub(" ", text))


_DIGITS = re.compile(r"\d+(?:[.,·]\d+)*")
_CITE = re.compile(r"\[(P\d{4}(?:\s*[,;]\s*P\d{4})*)\]")
_ABSTRACT = re.compile(r"abstract[- ]only|abstract (?:data|evidence|report)|(?:from|on) (?:the |an )?abstracts?\b", re.I)
_FULLTEXT = re.compile(r"full[- ]text", re.I)
_HIGH = re.compile(r"\bhigh(?:er)?(?: or unclear)? risk|\bhigh(?:-| )risk|risk of bias (?:was |is )?high", re.I)
_UNCLEAR = re.compile(r"unclear (?:or high )?(?:risk|bias)|risk of bias (?:was |is )?unclear", re.I)
_RETRO = re.compile(r"\bretrospective", re.I)
_PROSP = re.compile(r"\bprospective", re.I)


def segments(sentence: str) -> list[tuple[str, set[str]]]:
    """Découpe une phrase en propositions, chacune avec les identifiants cités juste après elle.

    « AUC 0.87 [P0015] and 0.89 [P0165] » -> [(« AUC 0.87 [P0015] », {P0015}), (« and 0.89 [P0165] », {P0165})].
    Un texte placé après la dernière citation est rattaché à la dernière proposition.
    """
    out: list[tuple[str, set[str]]] = []
    pos = 0
    for m in _CITE.finditer(sentence):
        text = sentence[pos:m.end()]
        ids = set(_ID.findall(m.group(1)))
        if out and not _ID.sub("", text).strip(" []\t,;"):  # « [P0031][P0040] » : citations accolées
            out[-1] = (out[-1][0] + text, out[-1][1] | ids)
        else:
            out.append((text, ids))
        pos = m.end()
    tail = sentence[pos:]
    if tail.strip(" .;,"):
        if out:
            out[-1] = (out[-1][0] + tail, out[-1][1])
        else:
            out.append((tail, set(_ID.findall(tail))))
    return out


def _attribute_problems(text: str, ids: set[str], facts: dict[str, dict]) -> list[str]:
    """Affirmations vérifiables sur le plan d'étude, le niveau de preuve et le risque de biais des études citées."""
    probs = []
    fs = {i: facts[i] for i in ids if i in facts}
    if not fs:
        return probs
    if _ABSTRACT.search(text) and not _FULLTEXT.search(text):
        bad = [i for i, f in fs.items() if f["level"] != "abstract_only"]
        if bad:
            probs.append(f"'abstract only' stated for {', '.join(sorted(bad))}, assessed on full text")
    if _FULLTEXT.search(text) and not _ABSTRACT.search(text):
        bad = [i for i, f in fs.items() if f["level"] != "full_text"]
        if bad:
            probs.append(f"'full text' stated for {', '.join(sorted(bad))}, assessed on the abstract only")
    high, unclear = bool(_HIGH.search(text)), bool(_UNCLEAR.search(text))
    allowed = {"high", "unclear"} if (high and unclear) else {"high"} if high else {"unclear"} if unclear else None
    if allowed:
        bad = [i for i, f in fs.items() if f["overall"] not in allowed]
        if bad:
            probs.append(f"risk of bias stated as {'/'.join(sorted(allowed))} for {', '.join(sorted(bad))}, "
                         f"actual: {', '.join(fs[i]['overall'] for i in sorted(bad))}")
    retro, prosp = bool(_RETRO.search(text)), bool(_PROSP.search(text))
    if retro != prosp:
        word = "retrospective" if retro else "prospective"
        bad = [i for i, f in fs.items() if f["design"] and word not in f["design"]]
        if bad:
            probs.append(f"'{word}' design stated for {', '.join(sorted(bad))}, actual: "
                         + "; ".join(f"{i} {fs[i]['design']}" for i in sorted(bad)))
    return probs


def check_paragraph(
    text: str, blocks: dict[str, str], facts: dict[str, dict] | None = None, allow_numbers: bool = True
) -> list[str]:
    """Problèmes d'un paragraphe (liste vide = contrôles réussis). Contrôles faits proposition par proposition :

    - identifiant cité qui n'est pas une étude incluse ;
    - nombre absent des données des études citées pour cette proposition (ou nombre sans aucune citation) ;
    - si `facts` est fourni : plan d'étude (rétrospectif/prospectif), niveau de preuve (résumé seul / texte
      intégral) et risque de biais affirmés pour une étude qui ne les a pas ;
    - paragraphe sans aucune citation ;
    - si `allow_numbers` est faux (synthèse sans chiffres) : tout chiffre dans le texte est signalé, les valeurs
      devant figurer uniquement dans les tableaux générés par le code.
    """
    problems: list[str] = []
    all_nums = set().union(*(_numbers(b) for b in blocks.values())) if blocks else set()
    cited_any = False
    for sent in split_sentences(text):
        for seg, ids in segments(sent):
            unknown = ids - set(blocks)
            if unknown:
                problems.append(f"unknown study ID(s): {', '.join(sorted(unknown))}")
            known = ids & set(blocks)
            cited_any |= bool(known)
            if facts and known:
                problems += _attribute_problems(seg, known, facts)
            if not allow_numbers:
                digits = _DIGITS.findall(_ID.sub(" ", seg))
                if digits:
                    problems.append(f"numbers written in the text ({', '.join(dict.fromkeys(digits))}): values belong in Table 3")
                continue
            nums = _numbers(seg)
            if not nums:
                continue
            if known:
                allowed_nums = set().union(*(_numbers(blocks[i]) for i in known))
                missing = nums - allowed_nums
                if missing:
                    problems.append(
                        f"number(s) {', '.join(sorted(missing))} not found in the data of {', '.join(sorted(known))}"
                    )
            else:
                missing = nums - all_nums
                problems.append(
                    "number(s) without citation" + (f" and not found in any study: {', '.join(sorted(missing))}" if missing else "")
                )
    if not cited_any:
        problems.append("no study cited")
    return list(dict.fromkeys(problems))


def study_facts(rec: EvidenceRecord) -> dict:
    fv = rec.extracted.get("study_design")
    design = fv.value.lower() if fv is not None and fv.found and fv.grounded else ""
    return {"level": rec.evidence_level, "overall": rec.overall_risk, "design": design}


# --------------------------------------------------------------------------- rédaction par le LLM
class SynthesisText(BaseModel):
    paragraphs: list[str] = Field(min_length=1, max_length=4, description="1 to 3 paragraphs of plain text")


SYSTEM_SYNTH = """You write one section of a systematic review manuscript (narrative synthesis, no meta-analysis).
Use ONLY the study data provided. Rules:
- Every sentence that reports a finding cites its source study ID(s) in square brackets right after the claim,
  for example [P0015] or [P0015, P0031]. Use only the IDs given in the data.
- Write NO numbers at all: no digits, percentages, AUC values, confidence intervals, sample sizes or years.
  The values are shown to the reader in Table 3. Describe results qualitatively and carefully: higher, lower,
  similar, improved, not reported, not compared. Write counts in words ("three studies"). Refer to Table 3
  for the values when useful.
- Before stating that something is higher or lower, check in the data WHICH group each value belongs to
  (AI alone, radiologists alone, radiologists with AI, radiologists without AI) and compare like with like.
  If the data do not give the comparison, say it was not reported.
- Do not mention any study, author, country or result that is not in the data. A value marked "not reported"
  or "not available" must not be guessed.
- Attribute design, evidence level (full text / abstract only) and risk of bias exactly as given for each study.
  When several IDs share one bracket, the statement must be true for EVERY one of them; otherwise cite them separately.
- Never swap values between groups (AI vs radiologists, with vs without AI, internal vs external data).
- Say when evidence comes from an abstract only or from a study at high risk of bias, when relevant.
- Neutral, cautious academic English. No headings, no bullet points, no markdown.
- 1 to 3 paragraphs, at most 130 words each."""

SECTIONS: dict[str, tuple[str, str]] = {
    "overview": (
        "Overview of included studies",
        "Describe the included studies as a group: designs (retrospective, prospective, reader studies), settings and "
        "countries, populations and sample sizes, AI systems and their roles (stand-alone, second reader, reading aid, "
        "triage), and comparators.",
    ),
    "standalone": (
        "Stand-alone AI compared with radiologists",
        "Summarise the diagnostic accuracy of AI used as a stand-alone reader compared with radiologists "
        "(sensitivity, specificity, AUC), study by study. Point out where the comparison is not reported.",
    ),
    "assisted": (
        "AI as a reading aid, second reader or triage tool",
        "Summarise the findings of studies where AI supports radiologists (reading with AI versus without, second "
        "reader, triage or workload reduction, recall rate, cancer detection rate, reading time).",
    ),
    "conclusion": (
        "Main findings and conclusions",
        "Write the main findings of the review and a cautious conclusion, taking into account the risk of bias "
        "of the studies, the number of studies with full text versus abstract only, and the heterogeneity of "
        "designs and outcomes. Do not claim more certainty than the data allow.",
    ),
}


CHECK_VERSION = "3-no-numbers"  # à incrémenter quand les contrôles changent : force la régénération du cache

REVISE = """The paragraphs below were written from the study data but failed automatic checks.
Rewrite them so that every problem listed is fixed, following all the rules. Keep what is correct.

Paragraphs:
{paragraphs}

Problems found:
{problems}"""


def _problems_of(paragraphs: list[str], checker) -> list[list[str]]:
    return [checker(p) for p in paragraphs]


def _synthesis(out_dir: Path, research_question: str, data: str, llm_factory, checker=None) -> dict[str, list[str]]:
    """Paragraphes par section, avec cache (report/synthesis.json) indexé par l'empreinte données + prompts.

    Si `checker` est fourni, un paragraphe qui échoue aux contrôles est renvoyé UNE fois au modèle avec la liste
    des problèmes (critique automatique) ; la version révisée n'est gardée que si elle a moins de problèmes.
    Le premier jet est conservé dans le cache (audit).
    """
    cache_file = out_dir / "synthesis.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    llm = None
    result: dict[str, list[str]] = {}
    for key, (title, task) in SECTIONS.items():
        sig = hashlib.sha256((SYSTEM_SYNTH + REVISE + CHECK_VERSION + task + research_question + data).encode()).hexdigest()[:12]
        if cache.get(key, {}).get("signature") == sig:
            result[key] = cache[key]["paragraphs"]
            continue
        llm = llm or llm_factory()
        user = (
            f"Review question: {research_question}\n\nSection to write: {title}\nTask: {task}\n\n"
            f"Data of the included studies (the only allowed source):\n{data}"
        )
        print(f"  [LLM] section « {title} »...", flush=True)
        out = structured_call(SynthesisText, SYSTEM_SYNTH, user, agent=f"synthesis_{key}", llm=llm)
        first = [" ".join(p.split()) for p in out.paragraphs if p.strip()]
        entry = {"signature": sig, "paragraphs": first, "revised": False}
        if checker:
            probs = _problems_of(first, checker)
            n_before = sum(map(len, probs))
            if n_before:
                listing = "\n".join(f"- paragraph {k + 1}: {'; '.join(pp)}" for k, pp in enumerate(probs) if pp)
                rev_user = user + "\n\n" + REVISE.format(paragraphs="\n\n".join(first), problems=listing)
                print(f"  [LLM] révision de « {title} » ({n_before} problème(s))...", flush=True)
                try:
                    rev = structured_call(SynthesisText, SYSTEM_SYNTH, rev_user, agent=f"synthesis_{key}_revision", llm=llm)
                    revised = [" ".join(p.split()) for p in rev.paragraphs if p.strip()]
                    n_after = sum(map(len, _problems_of(revised, checker)))
                    entry.update({"first_draft": first, "problems_before": n_before, "problems_after": n_after})
                    if n_after < n_before:
                        entry.update({"paragraphs": revised, "revised": True})
                except Exception as e:  # noqa: BLE001  la révision est facultative
                    log.warning("Révision de %s échouée : %s", key, e)
        result[key] = entry["paragraphs"]
        cache[key] = entry
        cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _render_paragraphs(paragraphs: list[str], blocks: dict[str, str], stats: dict, facts: dict | None = None) -> str:
    out = []
    for p in paragraphs:
        problems = check_paragraph(p, blocks, facts, allow_numbers=False)
        stats["paragraphs"] += 1
        out.append(p)
        if problems:
            stats["flagged"] += 1
            out.append("> ⚠ Automatic check failed: " + "; ".join(dict.fromkeys(problems)) + ". Verify against the evidence table.")
    return "\n\n".join(out)


# --------------------------------------------------------------------------- gabarits
def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _search_date(run_dir: Path) -> str:
    m = re.search(r"run_(\d{8})", run_dir.name)
    return datetime.strptime(m.group(1), "%Y%m%d").strftime("%d %B %Y") if m else "(date of the run)"


def _pct(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"


def methods_md(run_dir: Path, protocol: Protocol, cfg: dict, ev: dict) -> str:
    ident = _load(run_dir / "prisma_identification.json").get("identification", {})
    elig = _load(run_dir / "fulltext" / "prisma_eligibility.json")
    model = os.getenv("OLLAMA_MODEL") or cfg.get("llm", {}).get("model", "qwen3.5:9b")
    tool = cfg.get("appraisal", {}).get("tool", "QUADAS-2")
    groups = cfg.get("extraction", {}).get("groups") or DEFAULT_GROUPS
    fields = ", ".join(f["name"].replace("_", " ") for g in groups for f in g["fields"])
    sources = ", ".join(SOURCE_NAMES.get(s, s) for s in ident.get("per_source", {}))
    pico = protocol.pico
    dates = f"from {protocol.date_from or 'inception'} to {protocol.date_to or 'the search date'}"
    inc = "\n".join(f"- **I{i}.** {c}" for i, c in enumerate(protocol.inclusion_criteria, 1))
    exc = "\n".join(f"- **E{i}.** {c}" for i, c in enumerate(protocol.exclusion_criteria, 1))

    return f"""### Protocol and eligibility criteria

The review question was framed with the {protocol.framework} framework: *{protocol.research_question}* The protocol (question, PICO elements and eligibility criteria) was drafted by a planning agent and then reviewed, modified and approved by the review team before the search.

- **Population:** {pico.population}
- **Intervention:** {pico.intervention}
- **Comparator:** {pico.comparator or "standard radiologist interpretation (as stated in the review question)"}
- **Outcomes:** {pico.outcome}

Inclusion criteria:

{inc}

Exclusion criteria:

{exc}

Publication dates were restricted {dates}; records outside this range were excluded by a deterministic rule. Languages: {", ".join(protocol.languages)}.

### Information sources and search strategy

{len(ident.get("per_source", {}))} sources were searched on {_search_date(run_dir)}: {sources}. A search agent proposed concept blocks (keywords and MeSH terms) for the population and the intervention; the Boolean queries were then built deterministically by code for each source (the outcome block was deliberately left out to preserve recall). At most {ident.get("max_results_per_source", "n/a")} records were retrieved per source in this development run. The full queries are given in Appendix A. Duplicates were removed by DOI, then PMID, then normalised title.

### Selection process

Titles and abstracts were screened independently by two simulated reviewers ({model}, temperature 0, fixed seed) using the numbered criteria above: reviewer A was instructed to favour recall and reviewer B precision. Each reviewer wrote its justification before its decision. Disagreements where one reviewer excluded and the other did not were resolved by an arbiter agent, which could exclude only on explicit evidence. Records without an abstract were passed to full-text assessment rather than excluded. Agreement between reviewers was measured with Cohen's kappa on the pass/exclude decision.

Full-text eligibility was assessed with the same dual-reviewer and arbiter design on the full text when an open-access version could be retrieved (Europe PMC XML or open-access PDF), otherwise on the abstract. For each criterion, the relevant passages of the full text were retrieved with a hybrid search (BM25 and dense embeddings, reciprocal rank fusion). Every automatic exclusion and every uncertain ("maybe") decision was reviewed by a human; human decisions ({elig.get("human_decisions", 0)}) were recorded separately from the automatic decisions in an audit file.

### Data extraction

Data were extracted by an extraction agent from the full text (or abstract) for the following fields: {fields}. For every value the agent had to provide a supporting quote. Each value was checked automatically: the quote must exist word for word in the source, and every significant number in the value must appear in the source; values failing both checks were flagged as unverified. Values were then reviewed by a human, and {ev.get("human_fields", 0)} values were corrected (each correction is recorded with the original automatic value).

### Risk of bias assessment

Risk of bias was assessed with {tool} for studies with an available full text. A "low" or "high" judgement required a quote retrieved word for word in the text, otherwise the domain was set to "unclear"; studies assessed on the abstract only were rated "unclear" in all domains. Overall risk was "high" if any domain was at high risk, "low" only if all domains were at low risk, and "unclear" otherwise. The automatic judgements were reviewed by a human and {ev.get("human_domains", 0)} domain judgements were corrected.

### Synthesis methods

Because of the heterogeneity of designs, AI roles and outcome definitions, no meta-analysis was performed. Results were synthesised by direction of effect following the SWiM guideline (Synthesis Without Meta-analysis). For each study, the language model listed the head-to-head comparisons available in the verified extraction data (outcome, role of AI, label and value of the AI arm and of the comparator arm). Each comparison was checked by code: both values had to appear in the data line of that outcome, each directly after its group label, in different groups. The direction of effect was then computed by code (favouring AI, favouring the comparator, or no clear difference when the relative difference was below {ev.get('threshold', 0.02):.0%} of the comparator value), studies were counted by direction (a study with discordant comparisons for the same outcome was counted as mixed) and an exact sign test was computed. Outcomes considered: sensitivity, specificity, AUC, cancer detection rate, recall rate and reading time. All synthesis text was generated by code from the verified comparisons."""


def results_selection_md(flow: dict, run_dir: Path) -> str:
    i, s, e, inc = flow["identification"], flow["screening"], flow["eligibility"], flow["included"]
    screen = _load(run_dir / "screening" / "prisma_screening.json")
    ft = _load(run_dir / "fulltext" / "prisma_fulltext_retrieval.json")
    elig = _load(run_dir / "fulltext" / "prisma_eligibility.json")
    per = "; ".join(
        f"{SOURCE_NAMES.get(k, k)} {v} (of {i['databases_available'].get(k, '?')} available)" for k, v in i["databases"].items()
    )
    reasons = "; ".join(f"{r} (n = {n})" for r, n in sorted(e["reasons"].items(), key=lambda kv: -kv[1]))
    return f"""The searches identified {i['total_retrieved']} records ({per}). After removal of {i['duplicates_removed']} duplicates, {i['unique_records']} records were screened on title and abstract, of which {s['excluded']} were excluded and {s['passed']} were sought for full-text assessment. Agreement between the two screening reviewers was {_pct(screen.get('inter_reviewer_agreement'))} (Cohen's kappa {_pct(screen.get('cohens_kappa'))}); {screen.get('arbiter_calls', 0)} disagreements were resolved by the arbiter.

An open-access full text was retrieved for {ft.get('full_text', 'n/a')} of the {e['sought']} reports; {ft.get('abstract_only', 'n/a')} were assessed on the abstract only and {e['not_retrieved']} could not be retrieved. Of the {e['assessed']} reports assessed for eligibility (kappa {_pct(elig.get('cohens_kappa'))}), {e['excluded']} were excluded: {reasons}. **{inc['studies']} studies were included** ({inc['full_text']} assessed on full text and {inc['abstract_only']} on the abstract only). The flow of records is shown in Figure 1."""


def limitations_md(flow: dict, run_dir: Path, ev: dict) -> str:
    i, e, inc = flow["identification"], flow["eligibility"], flow["included"]
    ident = _load(run_dir / "prisma_identification.json").get("identification", {})
    return f"""- **Search size.** This development run retrieved at most {ident.get('max_results_per_source', 'n/a')} records per source ({i['total_retrieved']} in total), far fewer than the records available (e.g. {max(i['databases_available'].values()) if i['databases_available'] else 'n/a'} in one source); relevant studies are certainly missing and recall was not measured against a human reference.
- **Access to full texts.** Only open-access full texts were used: {e['not_retrieved']} of {e['sought']} reports could not be retrieved at all, and {inc['abstract_only']} of the {inc['studies']} included studies were assessed on the abstract only, with incomplete extraction and unassessable risk of bias.
- **Simulated reviewers.** Both reviewers are the same language model with different instructions; their agreement measures consistency, not accuracy. Human reviewers corrected eligibility decisions and {ev.get('human_fields', 0) + ev.get('human_domains', 0)} extraction or risk-of-bias items.
- **Verification limits.** Automatic checks prove that a quote or a number exists in the source, not that it is interpreted correctly (for example a value from a subgroup reported as the main result, or table columns attributed to the wrong reader).
- **Synthesis by direction of effect.** Vote counting ignores the size of effects and their statistical significance within studies, and the "no clear difference" threshold is arbitrary. Only comparisons whose values could be verified were used ({ev['swim']['n_valid']} accepted, {ev['swim']['n_rejected']} rejected): comparisons reported in prose without a group label next to each value were lost. Free-text synthesis by the language model was abandoned after it attributed results to the wrong study, the wrong group or the wrong direction while passing the number checks.
- **Risk of bias.** Most judgements were drafted automatically from retrieved passages; {ev.get('rob', {}).get('unclear', 0)} of {inc['studies']} studies have an overall "unclear" rating, mainly for lack of accessible information.
- **No meta-analysis.** Heterogeneous designs, AI roles and outcome definitions precluded pooling."""


def _initials(name: str) -> str:
    parts = name.replace(".", " ").split()
    if not parts:
        return ""
    k = len(parts) - 1
    while k > 0 and parts[k - 1].islower():  # particules : « Cornelis Jan van Rooden » -> « van Rooden CJ »
        k -= 1
    return " ".join(parts[k:]) + " " + "".join(p[0] for p in parts[:k] if p[0].isalpha())


def references_md(records: list[EvidenceRecord], papers: dict[str, Paper]) -> str:
    lines = []
    for r in records:
        p = papers.get(r.paper_id)
        if p is None:
            continue
        authors = ", ".join(_initials(a).strip() for a in p.authors[:6]) + (", et al" if len(p.authors) > 6 else "")
        doi = f" doi:{p.doi}" if p.doi else (f" {p.url}" if p.url else "")
        lines.append(f"- **[{r.paper_id}]** {authors or 'Anonymous'}. {p.title.rstrip('.')}. *{p.journal or 'n.p.'}*. {p.year or 'n.d.'}.{doi}")
    return "\n".join(lines)


def background_refs_md(refs: list[dict]) -> str:
    if not refs:
        return "_No background reference could be verified._"
    lines = []
    for r in refs:
        p = Paper.model_validate(r["paper"])
        authors = ", ".join(_initials(a).strip() for a in p.authors[:6]) + (", et al" if len(p.authors) > 6 else "")
        doi = f" doi:{p.doi}" if p.doi else (f" {p.url}" if p.url else "")
        lines.append(f"- **[{r['id']}]** {authors or 'Anonymous'}. {p.title.rstrip('.')}. *{p.journal or 'n.p.'}*. {p.year or 'n.d.'}.{doi}")
    return "\n".join(lines)


def citation_audit_md(result: dict) -> str:
    rows = ["| Ref | Statement in the introduction | Supporting sentence in the source abstract |", "|---|---|---|"]
    for r in result.get("references", []):
        for q in r["quotes"]:
            rows.append(f"| {r['id']} | {_cell_md(q['claim'])} | {_cell_md(q['quote'])} |")
    unresolved = [c for c in result.get("claims", []) if not str(c.get("status", "")).startswith("accepted")]
    tail = ""
    if unresolved:
        tail = "\n\nStatements left without a verified source:\n\n" + "\n".join(
            f"- {_cell_md(c['claim'])} — {c.get('status', '')}" for c in unresolved)
    return "\n".join(rows) + tail


def _cell_md(t: str) -> str:
    return " ".join(str(t).split()).replace("|", "/")


def appendix_md(run_dir: Path) -> str:
    strat = _load(run_dir / "search_strategy.json")
    out = []
    for q in strat.get("queries", []):
        out.append(f"**{SOURCE_NAMES.get(q['source'], q['source'])}**\n\n```text\n{q['query']}\n```")
    return "\n\n".join(out)


def _design_class(design: str) -> str:
    d = design.lower()
    if re.search(r"reader study|multi-?reader|mrmc|reader performance", d):  # pas « double-reader » (organisation du dépistage)
        return "reader study"
    if re.search(r"\bprospective", d):
        return "prospective study"
    if re.search(r"retrospective", d):
        return "retrospective study"
    return "design not reported" if not d else "other design"


def overview_md(records: list[EvidenceRecord], facts: dict[str, dict], swim_results: dict[str, dict]) -> str:
    """Vue d'ensemble des études, entièrement calculée (aucune prose libre du LLM)."""
    groups: dict[str, list[str]] = {}
    for r in records:
        groups.setdefault(_design_class(facts[r.paper_id]["design"]), []).append(r.paper_id)
    designs = "; ".join(f"{k} (n = {len(v)}) [{', '.join(v)}]" for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])))
    full = [r.paper_id for r in records if r.evidence_level == "full_text"]
    abst = [r.paper_id for r in records if r.evidence_level != "full_text"]
    roles: dict[str, set[str]] = {}
    for pid, res in swim_results.items():
        for c in res.get("valid", []):
            roles.setdefault(c["ai_role"], set()).add(pid)
    role_txt = "; ".join(f"{k} (n = {len(v)}) [{', '.join(sorted(v))}]" for k, v in sorted(roles.items())) or "none verified"
    return (
        f"{len(records)} studies were included. Designs (as extracted): {designs}. "
        f"{len(full)} studies were assessed on the full text [{', '.join(full)}] and {len(abst)} on the abstract only "
        f"[{', '.join(abst)}]. Roles of AI in the verified comparisons (a study can contribute to several): {role_txt}. "
        "Settings, populations, AI systems and comparators of each study are given in Table 1."
    )


# --------------------------------------------------------------------------- introduction (LLM, contrôlée)
REF = "[REFERENCE NEEDED]"


class IntroductionText(BaseModel):
    paragraphs: list[str] = Field(min_length=3, max_length=5, description="3 to 5 paragraphs of plain text")


SYSTEM_INTRO = """You draft the Introduction of a systematic review manuscript. The review team will add references.
Write 3 to 4 paragraphs in neutral academic English, in this order:
1. the health problem and why the screening or clinical context matters;
2. the intervention (here the technology or method being evaluated) and how it can be used;
3. what is uncertain or debated, and why a systematic review is needed (the gap);
4. the objective: restate the review question.
Strict rules:
- Write NO numbers, percentages, statistics or years (the team will add sourced figures).
- Do NOT cite any author, study, guideline or organisation by name, and never write a reference like "(Smith et al., 2020)".
- After EVERY sentence that states a background fact, write the marker [REFERENCE NEEDED] so the team can add a source.
- Do not anticipate the results of this review.
- No headings, no bullet points, no markdown."""

REVISE_INTRO = """Your draft failed automatic checks:
{problems}
Rewrite the full introduction fixing every problem and following all rules."""

_NUMS_IN_TEXT = re.compile(r"\d[\d.,%]*")
_AUTHOR_CITE = re.compile(r"\bet al\.?|\([A-Z][A-Za-z-]+,? (?:19|20)\d\d\)|\b(?:WHO|NICE|USPSTF|ACR|IARC)\b")


def check_introduction(paragraphs: list[str]) -> list[str]:
    """Problèmes de l'introduction (liste vide = contrôles réussis)."""
    problems = []
    for k, p in enumerate(paragraphs, 1):
        text = re.sub(r"\[R\d+\]", "", p.replace(REF, ""))
        numbers = list(dict.fromkeys(_NUMS_IN_TEXT.findall(text)))
        if numbers:
            problems.append(f"paragraph {k}: contains numbers ({', '.join(numbers)[:60]})")
        if _ID.search(text):
            problems.append(f"paragraph {k}: cites an included study ID")
        if _AUTHOR_CITE.search(text):
            problems.append(f"paragraph {k}: names an author, study or organisation as a reference")
        if k < len(paragraphs) and REF not in p:
            problems.append(f"paragraph {k}: background paragraph without any [REFERENCE NEEDED] marker")
    return problems


def write_introduction(out_dir: Path, protocol: Protocol, llm_factory) -> tuple[list[str], list[str]]:
    """Introduction rédigée par le LLM à partir du seul protocole (cache report/introduction.json).

    Une révision si les contrôles échouent ; la version gardée est celle qui a le moins de problèmes.
    Renvoie (paragraphes, problèmes restants)."""
    cache_file = out_dir / "introduction.json"
    pico = protocol.pico
    user = (f"Review title: {protocol.title}\nReview question: {protocol.research_question}\n"
            f"Population: {pico.population}\nIntervention: {pico.intervention}\n"
            f"Comparator: {pico.comparator or 'standard practice'}\nOutcomes: {pico.outcome}")
    sig = hashlib.sha256((SYSTEM_INTRO + REVISE_INTRO + user).encode()).hexdigest()[:12]
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("signature") == sig:
            return cached["paragraphs"], check_introduction(cached["paragraphs"])
    llm = llm_factory()
    print("  [LLM] introduction...", flush=True)
    first = [" ".join(p.split()) for p in structured_call(IntroductionText, SYSTEM_INTRO, user, agent="introduction", llm=llm).paragraphs]
    best, probs = first, check_introduction(first)
    entry = {"signature": sig, "first_draft": first, "problems_before": len(probs), "revised": False}
    if probs:
        print(f"  [LLM] révision de l'introduction ({len(probs)} problème(s))...", flush=True)
        try:
            rev = structured_call(IntroductionText, SYSTEM_INTRO,
                                  user + "\n\n" + REVISE_INTRO.format(problems="\n".join(f"- {x}" for x in probs)),
                                  agent="introduction_revision", llm=llm)
            revised = [" ".join(p.split()) for p in rev.paragraphs]
            rprobs = check_introduction(revised)
            entry["problems_after"] = len(rprobs)
            if len(rprobs) < len(probs):
                best, probs, entry["revised"] = revised, rprobs, True
        except Exception as e:  # noqa: BLE001  la révision est facultative
            log.warning("Révision de l'introduction échouée : %s", e)
    entry["paragraphs"] = best
    cache_file.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    return best, probs


def introduction_md(paragraphs: list[str], problems: list[str], protocol: Protocol) -> str:
    body = "\n\n".join(paragraphs)
    if problems:
        body += "\n\n> ⚠ Automatic check failed: " + "; ".join(problems) + "."
    return ("> *Drafted by the language model from the protocol only (no figures by design). Background statements were "
            "referenced by the citation agent: [R1], [R2]... are sources whose abstract contains a sentence supporting the "
            f"statement (Appendix B); statements still marked {REF} could not be matched to a verified source.*"
            f"\n\n{body}\n\n**Objective.** {protocol.research_question}")


# --------------------------------------------------------------------------- abstract (rédigé par l'agent à partir des données)
def abstract_md(flow: dict, run_dir: Path, protocol: Protocol, cfg: dict, records: list[EvidenceRecord], sw: dict) -> str:
    i, e, inc = flow["identification"], flow["eligibility"], flow["included"]
    ident = _load(run_dir / "prisma_identification.json").get("identification", {})
    names = [SOURCE_NAMES.get(k, k) for k in i["databases"]]
    sources = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    tool = cfg.get("appraisal", {}).get("tool", "QUADAS-2")
    rob: dict[str, int] = {}
    for r in records:
        rob[r.overall_risk] = rob.get(r.overall_risk, 0) + 1
    rob_txt = ", ".join(f"{n} {k}" for k, n in sorted(rob.items(), key=lambda kv: -kv[1]))
    counts = "; ".join(sw.get("outcome_counts", [])) or "no outcome was compared in more than one study"
    return f"""**Background and objective.** {protocol.research_question}

**Methods.** {sources} were searched on {_search_date(run_dir)} (at most {ident.get('max_results_per_source', 'n/a')} records per source in this development run). Records were screened on title and abstract and then on full text by two simulated reviewers (a local language model with two different instructions), with an arbiter for disagreements and human review of flagged decisions. Data were extracted with automatic verification against the source text; risk of bias was assessed with {tool}. Results were synthesised by direction of effect (SWiM) without meta-analysis.

**Results.** {i['total_retrieved']} records were identified and {i['unique_records']} remained after removal of duplicates; {e['sought']} reports were sought, {e['not_retrieved']} could not be retrieved and {e['assessed']} were assessed for eligibility. {inc['studies']} studies were included ({inc['full_text']} assessed on full text, {inc['abstract_only']} on the abstract only). Overall risk of bias: {rob_txt}. Verified head-to-head comparisons were available for {sw['studies_with_data']} studies ({sw['n_valid']} comparisons). Outcomes compared in at least two studies: {counts}.

**Conclusions.** The evidence gathered in this run is insufficient to conclude whether AI performs better or worse than radiologists: few studies provided verifiable comparisons, many were at high or unclear risk of bias, and the search was deliberately limited. The findings must be confirmed with a complete search and full-text access.

*Keywords: systematic review; PRISMA 2020; synthesis without meta-analysis (SWiM); AI-assisted evidence synthesis.*"""


# --------------------------------------------------------------------------- assemblage
def write_manuscript(run_dir: Path, protocol: Protocol, cfg: dict, llm_factory=get_llm) -> dict:
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    flow = collect_flow(run_dir)
    records, problems = current_records(run_dir)
    metas = {d["id"]: d for d in json.loads((run_dir / "fulltext" / "included_for_extraction.json").read_text(encoding="utf-8"))}
    papers = {k: Paper.model_validate(v) for k, v in metas.items()}

    rob_counts: dict[str, int] = {}
    for r in records:
        rob_counts[r.overall_risk] = rob_counts.get(r.overall_risk, 0) + 1
    ev = {
        "human_fields": sum(fv.human for r in records for fv in r.extracted.values()),
        "human_domains": sum(d.human for r in records for d in r.risk_of_bias),
        "rob": rob_counts,
    }

    from agents.swim import DEFAULT_THRESHOLD, extract_comparisons, render

    facts = {r.paper_id: study_facts(r) for r in records}
    threshold = float(cfg.get("synthesis", {}).get("no_difference_relative", DEFAULT_THRESHOLD))
    swim_results = extract_comparisons(records, out / "swim.json", llm_factory)
    sw = render(swim_results, facts, threshold)
    intro_paragraphs, intro_problems = write_introduction(out, protocol, llm_factory)
    from agents.citations import find_references, topic_anchors
    from tools.bibtex import to_bibtex

    strat = _load(run_dir / "search_strategy.json")
    pop_terms = [t for b in strat.get("blocks", []) if b.get("concept") == "population"
                 for t in b.get("keywords", []) + b.get("mesh_terms", [])] or [protocol.pico.population]
    cites = find_references(intro_paragraphs, protocol, out / "intro_references.json", llm_factory,
                            exclude_dois={p.doi for p in papers.values()}, anchors=topic_anchors(pop_terms),
                            topic_hint=pop_terms[0] if pop_terms else "")
    intro_paragraphs = cites["paragraphs"]
    (out / "references_background.bib").write_text(
        to_bibtex([Paper.model_validate(r["paper"]) for r in cites["references"]]), encoding="utf-8")
    from agents.prisma import render_png

    render_png(flow, out / "prisma_flow.png")
    ev["swim"], ev["threshold"] = sw, threshold

    rob_line = ", ".join(f"{n} {k}" for k, n in sorted(rob_counts.items()))
    md = f"""# {protocol.title.replace(": A Systematic Review Protocol", "").replace(" Protocol", "")}: a systematic review

{DISCLAIMER}

*Generated on {datetime.now():%d %B %Y} from `{run_dir.as_posix()}`. Reported following PRISMA 2020.*

## Abstract

{abstract_md(flow, run_dir, protocol, cfg, records, sw)}

## 1. Introduction

{introduction_md(intro_paragraphs, intro_problems, protocol)}

## 2. Methods

{methods_md(run_dir, protocol, cfg, ev)}

## 3. Results

### 3.1 Study selection

{results_selection_md(flow, run_dir)}

![Figure 1. PRISMA 2020 flow diagram of the selection of studies.](prisma_flow.png)

### 3.2 Characteristics of included studies

{overview_md(records, facts, swim_results)}

**Table 1. Characteristics of included studies.**

{characteristics_md(records)}

### 3.3 Risk of bias

Overall risk of bias ({cfg.get("appraisal", {}).get("tool", "QUADAS-2")}): {rob_line}. Judgements by domain are given in Table 2.

**Table 2. Risk of bias by domain (automatic draft reviewed by a human; † = corrected by a human).**

{rob_md(records)}

### 3.4 Results of individual studies

**Table 3. Diagnostic accuracy and other outcomes as reported by each study** (⚠ = value not verified in the text; † = corrected by a human).

{results_md(records)}

### 3.5 Synthesis by direction of effect (SWiM)

**Table 4. Direction of effect by study, AI role and outcome.** {sw['note']}

{sw['table']}

{sw['synthesis']}

Studies without a verifiable head-to-head comparison: {sw['without'] or 'none'}.

## 4. Discussion

### 4.1 Main findings

{sw['conclusion']}

### 4.2 Limitations

{limitations_md(flow, run_dir, ev)}

### 4.3 Comparison with other reviews and implications

> To be written by the review team.

## 5. Other information

Protocol, prompts, model outputs and all decisions (automatic and human) are stored in the run folder (`protocol.json`, `trace.jsonl`, `screening/`, `fulltext/`, `evidence/`). Model: {os.getenv("OLLAMA_MODEL") or cfg.get("llm", {}).get("model", "qwen3.5:9b")} via Ollama, temperature 0, fixed seed.

## References (background, introduction)

{background_refs_md(cites["references"])}

## References (included studies)

{references_md(records, papers)}

## Appendix A. Search queries

{appendix_md(run_dir)}

## Appendix B. Verification of background references

{citation_audit_md(cites)}
"""
    (out / "manuscript.md").write_text(md, encoding="utf-8")
    docx_path = None
    try:
        from tools.docx_export import markdown_to_docx

        docx_path = out / "manuscript.docx"
        try:
            markdown_to_docx(md, docx_path)
        except PermissionError:  # fichier ouvert dans Word : on n'écrase pas, on écrit à côté
            docx_path = out / f"manuscript_{datetime.now():%Y%m%d_%H%M%S}.docx"
            markdown_to_docx(md, docx_path, base_dir=out)
            problems.append(f"manuscript.docx is open in another program: saved as {docx_path.name}")
    except ImportError:
        log.warning("python-docx absent : manuscript.docx non généré (pip install python-docx).")
    return {
        "dir": str(out), "studies": len(records), "comparisons": sw["n_valid"], "rejected": sw["n_rejected"],
        "studies_with_data": sw["studies_with_data"], "introduction_problems": intro_problems,
        "background_refs": len(cites["references"]), "claims": len(cites["claims"]),
        "claims_resolved": sum(str(c.get("status", "")).startswith("accepted") for c in cites["claims"]),
        "docx": str(docx_path) if docx_path else None, "warnings": problems + flow["warnings"],
    }


def print_manuscript(res: dict) -> None:
    print("\n=== MANUSCRIT (BROUILLON) ===")
    print(f"  Markdown           : {res['dir']}\\manuscript.md")
    if res["docx"]:
        print(f"  Word               : {res['docx']}")
    print(f"  Études             : {res['studies']}")
    print(f"  Introduction       : {'contrôles OK' if not res['introduction_problems'] else '⚠ ' + '; '.join(res['introduction_problems'])}")
    print(f"  Références (intro) : {res['claims_resolved']}/{res['claims']} affirmations sourcées, "
          f"{res['background_refs']} références vérifiées (annexe B)")
    print(f"  Synthèse SWiM      : {res['comparisons']} comparaisons vérifiées dans {res['studies_with_data']} études "
          f"({res['rejected']} rejetées par les contrôles)")
    for w in res["warnings"]:
        print(f"  ATTENTION : {w}")
