import json
from pathlib import Path

import pytest

from trademesh.app.backend import graph_dot, highlight_citations, load_graph, load_state, load_vision, md_safe

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


def test_dollar_signs_are_escaped_so_markdown_does_not_render_latex():
    assert md_safe("$139,996 million and $54,052 million") == r"\$139,996 million and \$54,052 million"
    assert md_safe(r"already \$5") == r"already \$5"  # no double escaping
    assert highlight_citations("Revenue was $5 million [P1].") == r"Revenue was \$5 million **[P1]**."


def test_state_loads_fixture_with_warnings(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRADEMESH_DATA", str(tmp_path))          # empty dir: the fixture, even when a real corpus exists
    s = load_state()
    # Role B's ranking package is now discovered, so the dev stub is no longer used:
    # the remaining warnings are the fixture and the stub LLM.
    assert s.is_fixture and not s.using_stub and len(s.warnings) == 2
    assert {"bm25", "bm25+prf"} <= set(s.retrievers)


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_runs_every_tab_without_crashing(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception
    labels = {t.label for t in at.tabs}
    assert {"Ask", "Search", "Graph", "Pages", "Evaluation", "System", "Retrieval", "RAG answers", "Abstention"} <= labels
    text = _md(at)
    assert "tm-hero" in text and "tm-marquee" in text and "tm-foot" in text            # hero, marquee, footer are drawn
    assert "Page retrieval" in text and ("Page images (MaxSim)" in text or "No signal" in text)   # the Pages tab shows the demo, or says there is none; no crash
    at.text_input(key="ask_q").set_value("What does Tesla say about supplier concentration?").run()
    assert not at.exception and "tm-cite" in _md(at)
    at.text_input(key="search_q").set_value("Telsa battery").run()
    assert not at.exception
    at.text_input(key="typo_q").set_value("Micorsoft").run()
    assert not at.exception and "MSFT" in _md(at)
    at.toggle[0].set_value(True).run()   # closed-book toggle
    assert not at.exception


def test_call_to_action_buttons_fill_the_question(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert {"Ask with a typo", "Compare two companies"} <= {b.label for b in at.button}
    next(b for b in at.button if b.label == "Compare two companies").click().run()
    assert not at.exception and at.text_input(key="ask_q").value.startswith("Compare how Microsoft and Amazon")
    assert "MSFT, AMZN" in _md(at)                                                      # compare mode saw both companies


def test_ui_helpers_escape_and_render():
    from trademesh.app import ui
    from trademesh.types import Passage
    p = Passage("id<1>", "AAPL", "Apple", 2025, "1A", "text", "<script>alert(1)</script> costs $5 and $6 supplier risk")
    card = ui.passage_card("P1", p, 2.0, 4.0, cited=True, explain=[("supplier", 1.5)])
    assert "<script>" not in card and "&lt;script&gt;" in card and "cited" in card
    assert "$" not in card and "&#36;5" in card                          # dollar signs cannot start a LaTeX span
    assert "Why this score" in card and "supplier" in card
    assert ui.cite_html("Sales rose [P2].") == 'Sales rose <span class="tm-cite">P2</span>.'
    assert ui.mark_html("a **hit** b") == "a <mark>hit</mark> b"
    assert "width:50.0%" in ui.bar(0.5) and "width:100.0%" in ui.bar(7)     # clamped
    ok, bad, na = ui.checks([("x", True, "")]), ui.checks([("x", False, "d")]), ui.checks([("x", None, "")])
    assert "tm-check ok" in ok and "tm-check bad" in bad and "Flag" in bad and "N/A" in na
    assert "Other sections" in ui.passage_card("P1", Passage("i", "A", "A", 2025, "X", "text", "t"))


def test_every_icon_is_inline_svg_from_the_phosphor_set():
    from trademesh.app import ui
    from trademesh.app.icons import PATHS
    assert len(PATHS) >= 20
    for name, inner in PATHS.items():
        svg = ui.icon(name)
        assert svg.startswith("<svg") and 'viewBox="0 0 256 256"' in svg and "<path" in inner, name
        assert not any(bad in inner for bad in ("<script", "<style", "href", "onload")), name
    with pytest.raises(KeyError):
        ui.icon("no-such-icon")


def test_images_exist_and_are_credited():
    from trademesh.app import ui
    static = Path(ui.HERE) / "static"
    credits = ui.credits()
    assert {c["file"] for c in credits} == {"hero-wall-street.jpg", "nyse-sign.jpg", "manhattan-1931.jpg"}
    for c in credits:
        assert (static / c["file"]).stat().st_size > 50_000, c["file"]       # a real photograph, not an error page
        assert c["license"] and c["url"].startswith("https://commons.wikimedia.org/")
    assert "Image and icon credits" in ui.footer() and "Phosphor" in ui.footer()
    for name in ("hero-wall-street.jpg", "nyse-sign.jpg"):
        assert ui.img(name) in ui.hero()


def test_hero_marquee_tiles_and_pipeline_markup():
    from trademesh.app import ui
    hero = ui.hero()
    assert hero.count("<h1") == 1 and "tm-hero-img" in hero
    for banned in ("SECTION", "<button", "tm-badge"):                    # no meta-labels, tags or stats in the hero
        assert banned not in hero
    m = ui.marquee([("AAPL", "Apple Inc."), ("TSLA", "Tesla, Inc.")])
    assert m.count("AAPL") == 4 and 'class="track"' in m and ui.marquee([]) == ""
    assert "bento" in ui.tiles([("a", "1", "x", False)], bento=True) and "bento" not in ui.tiles([("a", "1", "x", False)])
    assert "<p class=d>" in ui.pipeline([("Verify", "Pass", "ok")]) and "<p class=d>" not in ui.pipeline([("Other", "x", "ok")])
    assert "no-such-image" in ui.empty_state("a", "b", "search", "no-such-image.jpg")


def test_css_defines_every_block_the_markup_uses():
    from trademesh.app import ui
    sheet = ui.css()
    for cls in (".tm-hero", ".tm-marquee", ".tm-pipe", ".tm-cell", ".tm-panel", ".tm-card", ".tm-check", ".tm-tile", ".tm-bar",
                ".tm-empty", ".tm-table", ".tm-foot", ".tm-chip", ".tm-hazard", ".tm-cite", ".tm-num", ".tm-row", ".bento"):
        assert cls in sheet, cls
    assert not any(ord(c) > 0x2000 for c in sheet)                       # no emoji or stray symbols in the stylesheet


def test_readiness_headline_numbers_and_entity_loader(monkeypatch, tmp_path):
    from trademesh.app import backend
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRADEMESH_DATA", str(tmp_path))                  # fixture
    rows = backend.readiness(backend.load_state())
    assert [r[0] for r in rows][:4] == ["Corpus", "Retrievers", "Reranker", "LLM"] and not rows[0][1] and not rows[3][1]
    assert backend.load_entity_eval(tmp_path / "missing.json") is None
    # no result files -> every tile is pending and shows no number
    monkeypatch.setattr(backend, "RESULT_DIRS", [tmp_path])
    monkeypatch.setattr(backend, "ENTITY_EVAL", tmp_path / "nope.json")
    monkeypatch.setattr(backend, "load_entity_eval", lambda path=tmp_path / "nope.json": None)
    tiles = backend.headline_numbers()
    assert len(tiles) == 4 and all(t[3] and t[1] == "-" for t in tiles)


def test_headline_numbers_come_from_the_real_result_files():
    from trademesh.app import backend
    rag, _ = backend.load_rag_study()
    ret, _ = backend.load_results()
    if rag is None or ret is None:
        pytest.skip("result files not present")
    tiles = {t[0]: t for t in backend.headline_numbers()}
    human = backend.load_human_scores()                                   # the human ratings win when they exist
    best = (human["configs"] if human else rag["scores"]["configs"])["best_rag"]["mean_rating"]
    assert tiles["Answer quality, best-RAG (0 to 2)"][1] == f"{best:.2f}"
    assert ("human" in tiles["Answer quality, best-RAG (0 to 2)"][2]) == bool(human)
    assert tiles["Unanswerable declined"][1] == "6/6"
    top = max(v["nDCG@10"] for v in ret["overall"].values())
    assert tiles["Best retrieval"][1] == f"{top:.3f}"


def test_results_loader_and_evaluation_tab(monkeypatch, tmp_path):
    import json

    from trademesh.app import backend
    # nothing there -> a message, not an exception
    monkeypatch.setattr(backend, "RESULT_DIRS", [tmp_path])
    assert backend.load_results()[0] is None
    # SYNTHETIC file, only to check that the loader and the tab render a results payload
    fake = {"n_needs": 2, "baseline": "bm25", "overall": {"bm25": {"P@5": .5, "nDCG@10": .6, "AP": .4}},
            "by_type": {"bm25": {"entity": {"nDCG@10": .6}}}, "pr_11pt": {"bm25": [1.0] * 11}}
    (tmp_path / "retrieval_dev.json").write_text(json.dumps(fake))
    data, why = backend.load_results()
    assert why == "" and data["split"] == "dev" and data["hypotheses"] == []
    (tmp_path / "retrieval_test.json").write_text(json.dumps(fake))
    assert backend.load_results()[0]["split"] == "test"            # the locked run wins over dev
    (tmp_path / "retrieval_test.json").write_text("{not json")
    assert backend.load_results()[0] is None                       # malformed: reported, never raised


def test_rag_study_loader_reports_missing_and_malformed(monkeypatch, tmp_path):
    from trademesh.app import backend
    monkeypatch.setattr(backend, "RESULT_DIRS", [tmp_path])
    data, why = backend.load_rag_study()
    assert data is None and "not available yet" in why
    (tmp_path / "rag_scores_test.json").write_text("{not json", encoding="utf-8")
    data, why = backend.load_rag_study()
    assert data is None and "could not read" in why
    (tmp_path / "rag_scores_test.json").write_text(json.dumps({"configs": {}}), encoding="utf-8")
    assert backend.load_rag_study()[0] is None


def test_rag_study_loader_reads_the_real_results_when_present():
    from trademesh.app import backend
    data, why = backend.load_rag_study()
    if data is None:
        pytest.skip(why)
    assert {"closed_book", "bm25_rag", "best_rag"} <= set(data["scores"]["configs"])


def test_example_questions_are_the_demo_cache_questions():
    """The example buttons must use questions from the study / demo-warm list, so they work offline when the cache is warm."""
    from eval.common import NEEDS, read_csv
    from eval.demo_warm import DEMO
    from trademesh.app.backend import EXAMPLES
    known = {n["need"] for n in read_csv(NEEDS / "needs.csv")} | {u["question"] for u in read_csv(NEEDS / "unanswerable.csv")}
    known |= {ref for _, ref, _ in DEMO}
    assert len(EXAMPLES) == 4
    for label, q in EXAMPLES:
        assert q in known, f"{label}: {q!r} is not a demo/study question, so it would not be cached"


def test_human_scores_loader_and_evaluation_panel(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    import shutil

    from streamlit.testing.v1 import AppTest
    from trademesh.app import backend

    monkeypatch.setattr(backend, "RESULT_DIRS", [tmp_path])
    assert backend.load_human_scores() is None
    (tmp_path / "rag_scores_human.json").write_text("{not json", encoding="utf-8")
    assert backend.load_human_scores() is None                                    # malformed: ignored, never raised
    for name in ("rag_scores_test.json", "rag_bootstrap_test.json", "abstention_eval.json", "abstention_scoreonly.json"):
        shutil.copy(backend.ROOT / "eval" / "out" / name, tmp_path / name)
    fake = {"split": "test", "rated_by": ["Sam"], "n_questions": 20,
            "configs": {"closed_book": {"mean_rating": 1.0}, "bm25_rag": {"mean_rating": 1.2}, "best_rag": {"mean_rating": 1.9}},
            "vs_model": {"Sam vs model pass AI": {"n": 60, "identical": 0.8, "kappa": 0.7}}}
    (tmp_path / "rag_scores_human.json").write_text(json.dumps(fake), encoding="utf-8")
    assert backend.load_human_scores()["rated_by"] == ["Sam"]
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cache"))
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception
    text = _md(at)
    assert "Human ratings (blind)" in text and "Rated by Sam" in text and "1.90" in text
    assert "Read this first" not in text                                          # the model-only notice is replaced


def test_headline_tile_names_the_number_of_human_raters(monkeypatch):
    from trademesh.app import backend
    cfgs = {"closed_book": {"mean_rating": 0.5}, "bm25_rag": {"mean_rating": 0.9}, "best_rag": {"mean_rating": 1.2}}
    for raters, phrase in ((["A"], "rated blind by one human"), (["A", "B"], "rated blind by 2 humans (mean)")):
        monkeypatch.setattr(backend, "load_human_scores", lambda r=raters: {"rated_by": r, "configs": cfgs})
        tile = {t[0]: t for t in backend.headline_numbers()}["Answer quality, best-RAG (0 to 2)"]
        assert phrase in tile[2] and tile[1] == "1.20"
