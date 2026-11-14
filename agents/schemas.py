"""Modèles de données (Pydantic) pour le protocole et la stratégie de recherche."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PICO(BaseModel):
    population: str
    intervention: str
    comparator: str = ""
    outcome: str
    study_design: str = ""


class Protocol(BaseModel):
    title: str
    research_question: str
    framework: str = "PICO"
    pico: PICO
    inclusion_criteria: list[str] = Field(min_length=1)
    exclusion_criteria: list[str] = Field(min_length=1)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    date_from: str = ""  # année, ex. "2015"
    date_to: str = ""


class ConceptBlock(BaseModel):
    concept: str  # population | intervention | outcome
    keywords: list[str] = Field(min_length=1)
    mesh_terms: list[str] = Field(default_factory=list)


class ConceptBlocks(BaseModel):
    blocks: list[ConceptBlock] = Field(min_length=1)


class SourceQuery(BaseModel):
    source: str
    query: str


class SearchStrategy(BaseModel):
    blocks: list[ConceptBlock]
    queries: list[SourceQuery]


class ScreeningDecision(BaseModel):
    # Ordre voulu : la justification est écrite AVANT la décision (aide un petit modèle à raisonner)
    rationale: str = Field(description="1-2 sentences citing specific details of the title/abstract")
    criterion: str = Field(description="Code of the main criterion, e.g. I2 or E3; 'none' for a clean include")
    decision: Literal["include", "exclude", "maybe"]
    confidence: Literal["low", "medium", "high"]


class ScreeningRecord(BaseModel):
    paper_id: str
    title: str
    stage: str = "title_abstract"
    evidence_level: str = ""  # full_text | abstract_only | none (utilisé à l'éligibilité)
    reviewer_a: ScreeningDecision | None = None
    reviewer_b: ScreeningDecision | None = None
    arbiter: ScreeningDecision | None = None
    final_decision: Literal["include", "exclude", "maybe"]
    final_criterion: str = ""
    final_rationale: str
    decided_by: Literal["rule", "consensus", "arbiter", "error", "human"]
    needs_human_review: bool = False


class ExtractedField(BaseModel):
    name: str
    evidence: str = Field(description="Exact quote (max 30 words) copied from the text; empty if not found")
    value: str = Field(description="Value as written in the text, or 'not reported'")
    found: bool


class ExtractionBatch(BaseModel):
    entries: list[ExtractedField]


class FieldValue(BaseModel):
    value: str
    evidence: str = ""
    found: bool = False
    evidence_verified: bool = False  # la citation existe mot pour mot dans le texte source
    numbers_verified: bool | None = None  # tous les nombres de la valeur figurent dans la source (None : aucun nombre)
    human: bool = False  # valeur saisie ou corrigée par un humain (evidence/human_overrides.csv)

    @property
    def grounded(self) -> bool:
        """Valeur étayée : citation exacte retrouvée, ou nombres tous retrouvés. Un nombre introuvable l'exclut toujours.
        Une valeur corrigée par un humain est considérée comme vérifiée (la note de l'humain sert de justification)."""
        if self.human:
            return self.found
        return self.found and self.numbers_verified is not False and (self.evidence_verified or self.numbers_verified is True)


class DomainJudgement(BaseModel):
    domain: str
    evidence: str = Field(description="Exact quote (max 30 words) from the text supporting the judgement; empty if none")
    rationale: str = Field(description="1-2 sentences")
    judgement: Literal["low", "high", "unclear"]


class AppraisalBatch(BaseModel):
    domains: list[DomainJudgement]


class DomainResult(BaseModel):
    domain: str
    judgement: Literal["low", "high", "unclear"]  # décision finale (après vérification de la citation)
    llm_judgement: Literal["low", "high", "unclear"]  # ce que le modèle avait proposé
    rationale: str
    evidence: str = ""
    evidence_verified: bool = False
    human: bool = False  # jugement corrigé par un humain (evidence/human_overrides.csv)


class EvidenceRecord(BaseModel):
    paper_id: str
    title: str
    year: int | None = None
    evidence_level: str
    eligibility: str = ""
    extracted: dict[str, FieldValue] = Field(default_factory=dict)
    risk_of_bias: list[DomainResult] = Field(default_factory=list)
    overall_risk: str = "unclear"
    tool: str = ""
    error: str = ""
