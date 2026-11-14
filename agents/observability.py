"""Observabilité d'un run : appels LLM par agent (durée, essais, erreurs), versions des prompts, empreintes des fichiers."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

KEY_FILES = [
    "protocol.json", "search_strategy.json", "records_deduplicated.json",
    "screening/decisions.jsonl", "screening/human_overrides.csv",
    "fulltext/eligibility.jsonl", "fulltext/human_overrides.csv", "fulltext/included_for_extraction.json",
    "evidence/records.jsonl", "evidence/human_overrides.csv",
    "report/prisma_flow.json", "report/swim.json", "report/manuscript.md",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def prompt_versions() -> dict[str, str]:
    """Empreinte de chaque prompt système (un changement de prompt est visible d'un run à l'autre)."""
    from agents import appraisal, eligibility, extraction, screening, swim

    prompts = {
        "screening_reviewer": screening.SYSTEM_REVIEWER + screening.STANCE_A + screening.STANCE_B,
        "screening_arbiter": screening.SYSTEM_ARBITER,
        "extraction": extraction.SYSTEM,
        "appraisal": appraisal.SYSTEM,
        "swim": swim.SYSTEM_SWIM + swim.EXAMPLE,
    }
    for name in ("SYSTEM_REVIEWER", "SYSTEM", "SYSTEM_ARBITER"):
        if hasattr(eligibility, name):
            prompts[f"eligibility_{name.lower()}"] = getattr(eligibility, name)
    return {k: hashlib.sha256(v.encode()).hexdigest()[:12] for k, v in prompts.items()}


def trace_summary(run_dir: Path) -> dict:
    """Appels LLM par agent à partir de trace.jsonl (réussites, erreurs, essais supplémentaires, secondes)."""
    path = run_dir / "trace.jsonl"
    per: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "retries": 0, "seconds": 0.0, "schema_echo": 0})
    if not path.exists():
        return {"agents": {}, "total_calls": 0, "total_errors": 0, "total_seconds": 0.0}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        agent = str(r.get("agent", "unknown"))
        agent = "eligibility" if agent.startswith("eligibility") else agent
        a = per[agent]
        if "error" in r:
            a["errors"] += 1
            a["schema_echo"] += "schema itself" in str(r["error"])
        else:
            a["calls"] += 1
            a["seconds"] += float(r.get("seconds") or 0)
            a["retries"] += int(r.get("attempt") or 1) > 1
    agents = {k: {**v, "seconds": round(v["seconds"], 1)} for k, v in sorted(per.items())}
    return {
        "agents": agents,
        "total_calls": sum(v["calls"] for v in agents.values()),
        "total_errors": sum(v["errors"] for v in agents.values()),
        "total_seconds": round(sum(v["seconds"] for v in agents.values()), 1),
    }


def run_summary(run_dir: Path) -> dict:
    files = {f: _sha(run_dir / f) for f in KEY_FILES if (run_dir / f).exists()}
    return {"llm": trace_summary(run_dir), "prompts": prompt_versions(), "files": files}


def summary_md(s: dict) -> str:
    rows = ["| Agent | Successful calls | Failed attempts | Calls needing a retry | LLM time (s) |", "|---|---|---|---|---|"]
    for k, v in s["llm"]["agents"].items():
        rows.append(f"| {k} | {v['calls']} | {v['errors']} | {v['retries']} | {v['seconds']} |")
    t = s["llm"]
    rows.append(f"| **total** | **{t['total_calls']}** | **{t['total_errors']}** | | **{t['total_seconds']}** |")
    prompts = ", ".join(f"{k} `{v}`" for k, v in s["prompts"].items())
    return "\n".join(rows) + f"\n\nPrompt versions (SHA-256 prefix): {prompts}."
