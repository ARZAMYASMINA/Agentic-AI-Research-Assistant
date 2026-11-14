"""Appraisal Agent : risque de biais par domaine (QUADAS-2, RoB 2), avec citation vérifiée.

Règles de prudence, appliquées par le code et non laissées au modèle :
- résumé seul -> tous les domaines « unclear » (sans appel au LLM) : on ne juge pas un biais sur un résumé ;
- un jugement « low » ou « high » exige une citation retrouvée mot pour mot dans le texte ;
  sinon il est ramené à « unclear » (le jugement proposé par le modèle reste enregistré dans llm_judgement).
"""
from __future__ import annotations

import re

from agents.llm import structured_call
from agents.schemas import AppraisalBatch, DomainJudgement, DomainResult
from tools.paper import Paper
from tools.quotes import verify_quote

TOOLS: dict[str, dict[str, str]] = {
    "QUADAS-2": {  # études de précision diagnostique
        "patient_selection": (
            "Could the selection of patients have introduced bias? Consider: consecutive or random sample, "
            "case-control design avoided, inappropriate exclusions avoided."
        ),
        "index_test": (
            "Could the conduct or interpretation of the index test (the AI algorithm or reading strategy) have "
            "introduced bias? Consider: interpreted without knowledge of the reference standard, threshold pre-specified."
        ),
        "reference_standard": (
            "Is the reference standard likely to classify the target condition correctly and was it interpreted "
            "without knowledge of the index test? Consider: biopsy, follow-up duration."
        ),
        "flow_and_timing": (
            "Could the patient flow have introduced bias? Consider: appropriate interval between index test and "
            "reference standard, same reference standard for all patients, all patients included in the analysis."
        ),
    },
    "RoB 2": {  # essais randomisés
        "randomization_process": "Was the allocation sequence random and concealed? Were baseline differences compatible with chance?",
        "deviations_from_intended_interventions": "Were participants and carers aware of the assignment, and were there deviations from the intended intervention?",
        "missing_outcome_data": "Were outcome data available for all or nearly all randomized participants?",
        "measurement_of_the_outcome": "Was the outcome measured appropriately, and were assessors blinded to the intervention?",
        "selection_of_the_reported_result": "Was the reported result selected from several analyses or outcome measures?",
    },
}

SYSTEM = """You assess risk of bias with the {tool} tool for a systematic review, using only the text provided.
For each domain answer with one judgement:
- "low": the text positively shows safeguards against bias in this domain;
- "high": the text shows a design or conduct feature that is likely to bias the results;
- "unclear": the text does not give enough information. Something not being mentioned is "unclear", never "high".
Limitations acknowledged by the authors count as evidence: a domain in which the authors admit a bias
(for example verification bias, thresholds set on the same data, exclusions) cannot be "low".
`evidence` is ONE contiguous passage (maximum 30 words) copied character for character from the text that
supports your judgement (no "..." joining pieces, no paraphrase).
A "low" or "high" judgement REQUIRES a non-empty `evidence`: put the supporting passage in `evidence`, not in
`rationale`. If you cannot copy such a passage from the text, the judgement must be "unclear" and `evidence` may be empty.
The domain descriptions in the user message are instructions, not part of the text: never quote them. `rationale` is 1-2 sentences. Return exactly one entry per domain, using the exact domain names."""

# Passages cités entre guillemets dans la justification (le modèle y met parfois la citation au lieu de `evidence`)
_QUOTED = re.compile(r"[\"“']([^\"“”']{20,400})[\"”']")


def recover_quote(evidence: str, rationale: str, source: str) -> tuple[str, bool, bool]:
    """(citation, vérifiée, récupérée depuis la justification).

    Si `evidence` est vide ou introuvable, on essaie les passages cités entre guillemets dans la justification :
    le premier retrouvé mot pour mot dans la source (au moins 4 mots) est retenu. Le critère de preuve reste le même.
    """
    evidence = evidence.strip()
    if evidence and verify_quote(evidence, source):
        return evidence, True, False
    for m in _QUOTED.finditer(rationale):
        cand = m.group(1).strip()
        if len(cand.split()) >= 4 and verify_quote(cand, source):
            return cand, True, True
    return evidence, False, False


def domain_queries(tool: str) -> list[str]:
    return [f"{name.replace('_', ' ')}: {desc}" for name, desc in TOOLS[tool].items()]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def appraise(paper: Paper, context: str, tool: str, source: str, llm) -> list[DomainResult]:
    domains = TOOLS[tool]
    lines = "\n".join(f"- {name}: {desc}" for name, desc in domains.items())
    user = f"Paper ID: {paper.id}\n\nText:\n{context}\n\nDomains to assess (one entry per domain, exact names):\n{lines}"
    batch = structured_call(AppraisalBatch, SYSTEM.format(tool=tool), user, agent="appraisal", llm=llm)
    return finalize_judgements(batch.domains, list(domains), source)


def finalize_judgements(raw: list[DomainJudgement], domain_names: list[str], source: str) -> list[DomainResult]:
    by_name = {_key(d.domain): d for d in raw}
    results: list[DomainResult] = []
    for name in domain_names:
        d = by_name.get(_key(name))
        if d is None:
            results.append(DomainResult(domain=name, judgement="unclear", llm_judgement="unclear", rationale="Domain not assessed by the model."))
            continue
        final, rationale = d.judgement, d.rationale.strip()
        evidence, verified, recovered = recover_quote(d.evidence, rationale, source)
        if recovered:
            rationale += " [Supporting quote recovered from the rationale.]"
        if d.judgement in ("low", "high") and not verified:
            final = "unclear"
            rationale += " [Downgraded to unclear: the supporting quote was not found in the text.]"
        results.append(
            DomainResult(
                domain=name, judgement=final, llm_judgement=d.judgement, rationale=rationale,
                evidence=evidence, evidence_verified=verified,
            )
        )
    return results


def abstract_only_results(tool: str, reason: str = "Only the abstract was available: risk of bias cannot be assessed.") -> list[DomainResult]:
    return [DomainResult(domain=n, judgement="unclear", llm_judgement="unclear", rationale=reason) for n in TOOLS[tool]]


def overall_risk(results: list[DomainResult]) -> str:
    """Un seul domaine à risque élevé suffit pour un risque global élevé ; un domaine incertain empêche « low »."""
    judgements = {r.judgement for r in results}
    if "high" in judgements:
        return "high"
    if "unclear" in judgements or not judgements:
        return "unclear"
    return "low"
