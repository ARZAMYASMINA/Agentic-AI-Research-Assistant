"""Planner Agent : transforme un sujet libre en protocole de revue systématique (PRISMA)."""
from __future__ import annotations

from agents.llm import structured_call
from agents.schemas import Protocol

SYSTEM = """You are an expert methodologist in systematic reviews (PRISMA 2020, Cochrane).
Given a topic, you write a rigorous, precise review protocol.
Rules:
- Use the PICO framework. For non-clinical topics, map it sensibly
  (population = subject/context, intervention = studied approach, outcome = measured result).
- research_question must be a single, answerable, focused question.
- Inclusion/exclusion criteria must be concrete and checkable from a title/abstract
  (study type, population, intervention, outcome, publication type, language, date).
- Give 4 to 7 inclusion criteria and 4 to 7 exclusion criteria.
- date_from / date_to are years (e.g. "2015"); leave date_to empty if open-ended.
- languages: a short list (usually ["en"]). Criteria must be consistent with it.
- Criteria must not contradict each other or the date range, and must not mention
  databases or search methods (those are handled separately).
- Write everything in English."""

USER = "Topic: {topic}\n\nWrite the full review protocol."


def generate_protocol(topic: str) -> Protocol:
    return structured_call(
        Protocol,
        system=SYSTEM,
        user=USER.format(topic=topic),
        agent="planner",
    )
