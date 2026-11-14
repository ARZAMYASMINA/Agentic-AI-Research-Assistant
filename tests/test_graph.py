import json

from agents.schemas import PICO, Protocol
from graph.pipeline import build_graph, mermaid, run_graph

PROTOCOL = Protocol(title="T", research_question="Q?", pico=PICO(population="p", intervention="i", outcome="o"),
                    inclusion_criteria=["inc"], exclusion_criteria=["exc"])


def _run(tmp_path, records=1):
    (tmp_path / "protocol.json").write_text(PROTOCOL.model_dump_json(), encoding="utf-8")
    (tmp_path / "search_strategy.json").write_text(json.dumps({"blocks": [], "queries": []}), encoding="utf-8")
    (tmp_path / "records_deduplicated.json").write_text(json.dumps([{"id": f"P{i}", "title": "t"} for i in range(records)]))
    (tmp_path / "index_info.json").write_text("{}")
    return tmp_path


def _patch(monkeypatch):
    import agents.review
    import agents.screening
    import agents.search_strategy

    monkeypatch.setattr(agents.search_strategy, "make_queries", lambda blocks, p, s: ([], []))
    monkeypatch.setattr(agents.screening, "run_screening", lambda run_dir, p: {"records_screened": 1, "passed_to_fulltext": 1})
    monkeypatch.setattr(agents.review, "review_queue", lambda run_dir, stage: ["x", "y"])


def test_graph_has_all_steps():
    m = mermaid({})
    for node in ("plan", "strategy", "retrieve", "screen", "review_screening", "eligibility", "extract", "manuscript", "critic"):
        assert node in m


def test_protocol_rejected_stops_graph(tmp_path, monkeypatch):
    _patch(monkeypatch)
    run = _run(tmp_path)
    asked = []
    state = run_graph(run, {}, ask=lambda q: asked.append(q) or "non", out=lambda *_: None)
    assert state["stop_reason"] == "protocol rejected by the human reviewer"
    assert len(asked) == 1 and not (run / "protocol_approved.txt").exists()


def test_two_human_interrupts_and_stop_at_review(tmp_path, monkeypatch):
    _patch(monkeypatch)
    run = _run(tmp_path)
    answers = iter(["oui", "s"])
    questions = []
    state = run_graph(run, {}, ask=lambda q: questions.append(q) or next(answers), out=lambda *_: None)
    assert (run / "protocol_approved.txt").exists()
    assert "2 décisions de screening" in questions[1]
    assert state["stop_reason"].startswith("stopped by the human reviewer at screening")
    assert any(l.startswith("screening: 1 screened") for l in state["log"])


def test_review_now_calls_reviewer_then_reruns_step(tmp_path, monkeypatch):
    _patch(monkeypatch)
    run = _run(tmp_path)
    (run / "protocol_approved.txt").write_text("ok")
    calls = []
    import agents.fulltext

    def stop_here(run_dir):
        raise RuntimeError("fulltext reached")

    monkeypatch.setattr(agents.fulltext, "run_fulltext_retrieval", stop_here)
    (run / "screening").mkdir()
    (run / "screening" / "included_for_fulltext.json").write_text("[]")
    graph = build_graph({}, review_fn=lambda rd, stage, p: calls.append(stage) or {"include": 1})
    try:
        run_graph(run, {}, ask=lambda q: "r", out=lambda *_: None, graph=graph)
    except RuntimeError as e:
        assert "fulltext reached" in str(e)
    assert calls == ["screening"]


def test_empty_search_routes_to_end(tmp_path, monkeypatch):
    _patch(monkeypatch)
    run = _run(tmp_path, records=0)
    state = run_graph(run, {}, auto=True, out=lambda *_: None)
    assert state["stop_reason"] == "no record found"
