import json
from pathlib import Path

import pytest

from trademesh.app.backend import graph_dot, highlight_citations, load_graph, load_state, load_vision

APP = Path(__file__).resolve().parents[1] / "src" / "trademesh" / "app" / "streamlit_app.py"


def test_missing_data_is_reported_not_raised(tmp_path):
    graph, why = load_graph(tmp_path / "nope.json")
    assert graph is None and "not available yet" in why
    vision, why = load_vision(tmp_path / "nope.json")
    assert vision is None and "not available yet" in why


def test_malformed_data_is_reported_not_raised(tmp_path):
    bad = tmp_path / "g.json"
    bad.write_text("{not json", encoding="utf-8")
    graph, why = load_graph(bad)
    assert graph is None and "could not read" in why
    bad.write_text(json.dumps({"nodes": [{"ticker": "A"}], "edges": []}), encoding="utf-8")   # no pagerank
    assert load_graph(bad)[0] is None


def test_graph_loads_and_renders_dot(tmp_path):
    g = {"nodes": [{"ticker": "AAPL", "pagerank": 0.6}, {"ticker": "TSLA", "pagerank": 0.4}],
         "edges": [{"source": "AAPL", "target": "TSLA", "weight": 2}, {"source": "AAPL", "target": "XXX"}]}
    p = tmp_path / "g.json"
    p.write_text(json.dumps(g), encoding="utf-8")
    graph, why = load_graph(p)
    assert graph and not why
    dot = graph_dot(graph)
    assert '"AAPL" -> "TSLA"' in dot and "XXX" not in dot


def test_highlight_citations():
    assert highlight_citations("Sales rose [P1][P2].") == "Sales rose **[P1]****[P2]**."


def test_state_loads_fixture_with_warnings(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    s = load_state()
    # Role B's ranking package is now discovered, so the dev stub is no longer used:
    # the remaining warnings are the fixture and the stub LLM.
    assert s.is_fixture and not s.using_stub and len(s.warnings) == 2
    assert {"bm25", "bm25+prf"} <= set(s.retrievers)


def test_app_runs_every_tab_without_crashing(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 4
    at.text_input(key="ask_q").set_value("What does Tesla say about supplier concentration?").run()
    assert not at.exception
    assert any("Answer" in s.value for s in at.subheader)
    at.text_input(key="search_q").set_value("Telsa battery").run()
    assert not at.exception
    # graph and pages tabs show the 'not available yet' message instead of failing
    assert any("not available yet" in i.value for i in at.info)
    at.toggle[0].set_value(True).run()   # closed-book toggle
    assert not at.exception
