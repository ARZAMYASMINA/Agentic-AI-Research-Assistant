"""Extraction Agent : champs structurés, chacun avec une citation exacte vérifiée contre le texte source.

Les champs sont définis par groupes (config YAML `extraction.groups`, sinon DEFAULT_GROUPS ci-dessous) :
un appel au LLM par groupe et par article, avec uniquement les passages pertinents du texte (RAG).
"""
from __future__ import annotations

import re

from agents.llm import structured_call
from agents.schemas import ExtractedField, ExtractionBatch, FieldValue
from tools.paper import Paper
from tools.quotes import numbers_in_source, verify_quote

DEFAULT_GROUPS: list[dict] = [
    {
        "name": "study",
        "fields": [
            {"name": "study_design", "description": "Study design (RCT, prospective cohort, retrospective cohort, reader study...)"},
            {"name": "country_and_setting", "description": "Country, screening programme or institution(s), data collection period"},
            {"name": "population", "description": "Participants: who was screened, age range or mean age, number of women and of examinations"},
            {"name": "number_of_cancers", "description": "Number of cancers (or interval cancers) in the study data"},
            {"name": "ai_system", "description": "Name, vendor or architecture of the AI system and its role (stand-alone, second reader, triage...)"},
            {"name": "comparator", "description": "Comparator: radiologist reading (single or double reading, experience level)"},
            {"name": "reference_standard", "description": "How cancers were confirmed (biopsy, follow-up) and follow-up duration"},
        ],
    },
    {
        "name": "results",
        "fields": [
            {"name": "sensitivity", "description": "Sensitivity of the AI, of the radiologists and of the combination, as reported (with confidence intervals)"},
            {"name": "specificity", "description": "Specificity of the AI, of the radiologists and of the combination, as reported (with confidence intervals)"},
            {"name": "auc", "description": "Area under the ROC curve (AUC) of the AI and of the radiologists"},
            {"name": "other_outcomes", "description": "Other outcomes: recall rate, cancer detection rate, false positives, workload reduction, reading time"},
            {"name": "main_conclusion", "description": "Main conclusion of the authors about the AI performance"},
            {"name": "stated_limitations", "description": "Limitations acknowledged by the authors"},
        ],
    },
]

SYSTEM = """You are a data extractor for a systematic review.
Extract the requested fields strictly from the text provided (title, abstract and excerpts).
Rules:
- Return exactly ONE entry per requested field, with the exact field name given. Never split a field into
  several entries and never add anything to the field name.
- Never invent or infer. If a field is not stated in the text, set found=false, value="not reported", evidence="".
- If found=true, give as `evidence` ONE contiguous passage (maximum 30 words) copied character for character
  from the text: no paraphrase, no "..." to join pieces, no reformatting. Quote the passage even when the
  value is a number. If the numbers come from a table and no single passage fits, `evidence` may be empty,
  but then every number of `value` must appear exactly as written in the text.
  Never compute, round or derive numbers (for example never convert percentages into counts).
- `value` contains only what the text states, with numbers, units and confidence intervals exactly as written.
  No comments, no interpretation, no remarks about inconsistencies.
  When results are given for several readers or strategies (AI, radiologists, combined), keep them separate
  inside the single `value`, for example "AI: ...; radiologists: ...".

Output format reminder: one entry per field, exact names."""

NOT_REPORTED = {"", "not reported", "not stated", "n/a", "na", "none", "unknown"}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _base(name: str) -> str:
    """« sensitivity: AI (AIsens) » -> « sensitivity »."""
    return re.split(r"[:(]", name, maxsplit=1)[0]


def _label(name: str) -> str:
    """« sensitivity: AI (AIsens) » -> « AI (AIsens) » ; « auc (AI) » -> « AI »."""
    if ":" in name:
        return name.split(":", 1)[1].strip()
    m = re.search(r"\((.*)\)", name)
    return m.group(1).strip() if m else ""


def field_queries(group: dict) -> list[str]:
    return [f"{f['name'].replace('_', ' ')}: {f['description']}" for f in group["fields"]]


def extract_group(paper: Paper, level: str, context: str, group: dict, llm) -> ExtractionBatch:
    spec_lines = "\n".join(f"- {f['name']}: {f['description']}" for f in group["fields"])
    note = "" if level == "full_text" else "Only the title and abstract are available: many fields may be 'not reported'.\n\n"
    user = (
        f"{note}Paper ID: {paper.id}\n\nText:\n{context}\n\n"
        f"Fields to extract (one entry per field, using these exact names):\n{spec_lines}"
    )
    return structured_call(ExtractionBatch, SYSTEM, user, agent="extraction", llm=llm)


def finalize_fields(raw: list[ExtractedField], specs: list[dict], source: str) -> dict[str, FieldValue]:
    """Un FieldValue par champ demandé.

    - Les entrées dont le nom commence par le nom du champ (« sensitivity: AI », « sensitivity: Radiologists »)
      sont fusionnées : le modèle découpe souvent un champ par lecteur, ces données ne doivent pas être perdues.
    - Champ sans aucune valeur -> « not reported ».
    - La valeur est « citation vérifiée » seulement si CHAQUE morceau a une citation retrouvée dans la source.
    """
    out: dict[str, FieldValue] = {}
    for spec in specs:
        skey = _key(spec["name"])
        group = [e for e in raw if _key(e.name) == skey or _key(_base(e.name)) == skey]
        good = [e for e in group if e.found and e.value.strip().lower() not in NOT_REPORTED]
        if not good:
            out[spec["name"]] = FieldValue(value="not reported")
            continue

        parts = []
        for e in good:
            label = _label(e.name)
            parts.append(f"{label}: {e.value.strip()}" if label else e.value.strip())
        value = "; ".join(parts)

        evidences: list[str] = []
        for e in good:
            ev = e.evidence.strip()
            if ev and ev not in evidences:
                evidences.append(ev)

        out[spec["name"]] = FieldValue(
            value=value,
            evidence=" | ".join(evidences),
            found=True,
            evidence_verified=all(e.evidence.strip() and verify_quote(e.evidence, source) for e in good),
            numbers_verified=numbers_in_source(value, source),
        )
    return out
