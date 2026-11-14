"""Orchestration LangGraph du pipeline PRISMA (Coordinator).

Graphe d'état : protocole -> stratégie -> récupération -> index -> screening -> relecture humaine -> textes intégraux
-> éligibilité -> relecture humaine -> extraction & biais -> rapport -> manuscrit -> agent critique.

- Chaque nœud appelle une étape existante qui sait reprendre depuis ses fichiers : le dossier du run est l'état
  durable (relancer le graphe sur un run existant ne refait que ce qui manque, sans rappeler le LLM pour le reste).
- Points de supervision humaine par `interrupt()` : validation du protocole, puis décision sur les relectures en
  attente (relire maintenant / plus tard / arrêter) après le screening et après l'éligibilité.
- Routage conditionnel : arrêt si la recherche ne trouve rien, si aucune étude n'est incluse, si le protocole est
  refusé, ou si l'agent critique trouve des erreurs.
"""
from __future__ import annotations

import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ReviewState(TypedDict, total=False):
    run_dir: str
    topic: str
    max_results: int
    auto: bool  # True : aucune interruption (validations sautées, relectures reportées)
    stop_reason: str
    log: Annotated[list[str], operator.add]  # journal des étapes (concaténé par LangGraph)


def _protocol(run_dir: Path):
    from agents.schemas import Protocol

    return Protocol.model_validate_json((run_dir / "protocol.json").read_text(encoding="utf-8"))


def build_graph(cfg: dict, review_fn: Callable | None = None):
    """Construit et compile le graphe. `review_fn(run_dir, stage, protocol)` = relecture humaine interactive."""
    if review_fn is None:
        from agents.review import interactive_review as review_fn  # noqa: N806

    def plan(state: ReviewState) -> dict:
        from agents.planner import generate_protocol

        run_dir = Path(state["run_dir"])
        pfile, approved = run_dir / "protocol.json", run_dir / "protocol_approved.txt"
        if not pfile.exists():
            pfile.write_text(generate_protocol(state["topic"]).model_dump_json(indent=2), encoding="utf-8")
        if approved.exists() or state.get("auto"):
            return {"log": ["plan: protocol loaded"]}
        p = _protocol(run_dir)
        answer = interrupt({
            "type": "approve_protocol",
            "question": f"Valider le protocole ? Vous pouvez d'abord modifier {pfile} puis répondre 'oui'. [oui/non]",
            "protocol": p.model_dump(),
        })
        if str(answer).strip().lower() not in ("o", "oui", "y", "yes"):
            return {"stop_reason": "protocol rejected by the human reviewer", "log": ["plan: rejected"]}
        _protocol(run_dir)  # relu : les modifications humaines du fichier doivent rester valides
        approved.write_text(f"approved {datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")
        return {"log": ["plan: protocol approved by human"]}

    def strategy(state: ReviewState) -> dict:
        from agents.schemas import SearchStrategy
        from agents.search_strategy import build_search_strategy, make_queries

        run_dir = Path(state["run_dir"])
        p, sources = _protocol(run_dir), cfg.get("search", {}).get("sources") or None
        sfile = run_dir / "search_strategy.json"
        if sfile.exists():
            s = SearchStrategy.model_validate_json(sfile.read_text(encoding="utf-8"))
            blocks, queries = make_queries(s.blocks, p, sources)  # requêtes reconstruites sans LLM
            s = SearchStrategy(blocks=blocks, queries=queries)
        else:
            s = build_search_strategy(p, sources)
        sfile.write_text(s.model_dump_json(indent=2), encoding="utf-8")
        return {"log": [f"strategy: {len(s.queries)} queries"]}

    def retrieve(state: ReviewState) -> dict:
        from agents.retrieval import run_retrieval
        from agents.schemas import SearchStrategy

        run_dir = Path(state["run_dir"])
        if not (run_dir / "records_deduplicated.json").exists():
            s = SearchStrategy.model_validate_json((run_dir / "search_strategy.json").read_text(encoding="utf-8"))
            run_retrieval(s, _protocol(run_dir), run_dir, state.get("max_results") or 200)
        n = len(json.loads((run_dir / "records_deduplicated.json").read_text(encoding="utf-8")))
        return {"log": [f"retrieve: {n} unique records"]} if n else {"stop_reason": "no record found", "log": ["retrieve: 0"]}

    def index(state: ReviewState) -> dict:
        run_dir = Path(state["run_dir"])
        if not (run_dir / "index_info.json").exists():
            from rag.index import build_index

            build_index(run_dir)
        return {"log": ["index: ready"]}

    def screen(state: ReviewState) -> dict:
        from agents.screening import run_screening

        s = run_screening(Path(state["run_dir"]), _protocol(Path(state["run_dir"])))
        return {"log": [f"screening: {s['records_screened']} screened, {s['passed_to_fulltext']} passed"]}

    def _review_node(stage: str, rerun: Callable[[Path], dict]):
        def node(state: ReviewState) -> dict:
            from agents.review import review_queue

            run_dir = Path(state["run_dir"])
            pending = len(review_queue(run_dir, stage))
            if not pending or state.get("auto"):
                return {"log": [f"review_{stage}: {pending} pending (not reviewed now)"]}
            answer = str(interrupt({
                "type": "human_review", "stage": stage, "pending": pending,
                "question": f"{pending} décisions de {stage} sont à relire. [r]elire maintenant / [p]lus tard / [s]top",
            })).strip().lower()[:1]
            if answer == "s":
                return {"stop_reason": f"stopped by the human reviewer at {stage} review", "log": [f"review_{stage}: stop"]}
            if answer == "r":
                counts = review_fn(run_dir, stage, _protocol(run_dir))
                rerun(run_dir)  # applique les décisions humaines (reprise, sans LLM)
                return {"log": [f"review_{stage}: {counts}"]}
            return {"log": [f"review_{stage}: postponed ({pending} pending)"]}
        return node

    def fulltext(state: ReviewState) -> dict:
        from agents.fulltext import run_fulltext_retrieval

        run_dir = Path(state["run_dir"])
        manifest = run_dir / "fulltext" / "manifest.json"
        wanted = {d["id"] for d in json.loads((run_dir / "screening" / "included_for_fulltext.json").read_text(encoding="utf-8"))}
        pdfs = list((run_dir / "fulltext" / "pdf").glob("*.pdf")) if (run_dir / "fulltext" / "pdf").exists() else []
        if manifest.exists():
            have = {m["id"] for m in json.loads(manifest.read_text(encoding="utf-8"))}
            if wanted <= have and all(p.stat().st_mtime <= manifest.stat().st_mtime for p in pdfs):
                return {"log": ["fulltext: up to date"]}
        r = run_fulltext_retrieval(run_dir)
        return {"log": [f"fulltext: {r.get('full_text')} full texts"]}

    def eligibility(state: ReviewState) -> dict:
        from agents.eligibility import run_eligibility

        run_dir = Path(state["run_dir"])
        e = run_eligibility(run_dir, _protocol(run_dir))
        n = e["included"] + e["maybe_human_decision"]
        return {"log": [f"eligibility: {n} studies kept"]} if n else {"stop_reason": "no study included", "log": ["eligibility: 0"]}

    def extract(state: ReviewState) -> dict:
        from agents.evidence import run_extraction

        run_dir = Path(state["run_dir"])
        s = run_extraction(run_dir, _protocol(run_dir), cfg)
        return {"log": [f"extraction: {s['studies']} studies, grounded {s['grounded_rate']}"]}

    def report(state: ReviewState) -> dict:
        from agents.evidence import refresh_evidence_tables
        from agents.report import write_report_data

        run_dir = Path(state["run_dir"])
        refresh_evidence_tables(run_dir, cfg)
        r = write_report_data(run_dir)
        return {"log": [f"report: {r['studies']} studies, {r['references']} references"]}

    def manuscript(state: ReviewState) -> dict:
        from agents.manuscript import write_manuscript

        run_dir = Path(state["run_dir"])
        m = write_manuscript(run_dir, _protocol(run_dir), cfg)
        return {"log": [f"manuscript: {m['comparisons']} verified comparisons"]}

    def critic(state: ReviewState) -> dict:
        from agents.critic import run_qa

        run_dir = Path(state["run_dir"])
        q = run_qa(run_dir, _protocol(run_dir))
        c = q["counts"]
        upd: dict = {"log": [f"critic: {c['error']} errors, {c['warning']} warnings"]}
        if c["error"]:
            upd["stop_reason"] = f"quality check found {c['error']} errors (see report/qa_report.md)"
        return upd

    def go_on(state: ReviewState) -> str:
        return "stop" if state.get("stop_reason") else "next"

    def _rerun_screening(run_dir: Path) -> dict:
        from agents.screening import run_screening

        return run_screening(run_dir, _protocol(run_dir))

    def _rerun_eligibility(run_dir: Path) -> dict:
        from agents.eligibility import run_eligibility

        return run_eligibility(run_dir, _protocol(run_dir))

    g = StateGraph(ReviewState)
    steps = [
        ("plan", plan), ("strategy", strategy), ("retrieve", retrieve), ("index", index), ("screen", screen),
        ("review_screening", _review_node("screening", _rerun_screening)), ("fulltext", fulltext),
        ("eligibility", eligibility), ("review_eligibility", _review_node("eligibility", _rerun_eligibility)),
        ("extract", extract), ("report", report), ("manuscript", manuscript), ("critic", critic),
    ]
    for name, fn in steps:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    for (a, _), (b, _) in zip(steps, steps[1:]):
        g.add_conditional_edges(a, go_on, {"next": b, "stop": END})
    g.add_edge("critic", END)
    return g.compile(checkpointer=MemorySaver())


def run_graph(run_dir: Path, cfg: dict, topic: str = "", max_results: int = 200, auto: bool = False,
              ask: Callable[[str], str] = input, out: Callable[[str], None] = print, graph=None) -> dict:
    """Exécute le graphe ; chaque interruption est posée à l'humain dans le terminal, puis le graphe reprend."""
    graph = graph or build_graph(cfg)
    config = {"configurable": {"thread_id": str(run_dir)}}
    payload: object = {"run_dir": str(run_dir), "topic": topic, "max_results": max_results, "auto": auto, "log": []}
    while True:
        state = graph.invoke(payload, config)
        pending = state.get("__interrupt__")
        if not pending:
            break
        req = pending[0].value
        if req.get("type") == "approve_protocol":
            p = req["protocol"]
            out(f"\nProtocole : {p['title']}\nQuestion : {p['research_question']}")
            out("\n".join(f"  I{i}. {c}" for i, c in enumerate(p["inclusion_criteria"], 1)))
            out("\n".join(f"  E{i}. {c}" for i, c in enumerate(p["exclusion_criteria"], 1)))
        payload = Command(resume=ask(f"\n[supervision humaine] {req['question']} : "))
    for line in state.get("log", []):
        out(f"  - {line}")
    out(f"\nArrêt : {state['stop_reason']}" if state.get("stop_reason") else "\nGraphe terminé.")
    return state


def mermaid(cfg: dict) -> str:
    return build_graph(cfg, review_fn=lambda *a: {}).get_graph().draw_mermaid()
