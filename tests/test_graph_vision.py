import json

import numpy as np
import pytest

from trademesh.graph.mentions import build_graph, count_mentions, hits, pagerank
from trademesh.types import Passage
from trademesh.vision.maxsim import maxsim, rank_pages, write_results


def test_pagerank_two_node_hand_values():
    # a -> b, b dangling: x = 0.075 + 0.425 y and x + y = 1  =>  x = 0.5/1.425 = 0.350877, y = 0.649123
    pr = pagerank(["a", "b"], [("a", "b")])
    assert pr["a"] == pytest.approx(0.350877, abs=1e-5) and pr["b"] == pytest.approx(0.649123, abs=1e-5)
    assert sum(pr.values()) == pytest.approx(1.0)


def test_pagerank_symmetric_cycle_is_uniform_and_hub_wins():
    cyc = pagerank(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")])
    assert all(v == pytest.approx(1 / 3) for v in cyc.values())
    star = pagerank(list("abcd"), [("b", "a"), ("c", "a"), ("d", "a")])
    assert max(star, key=star.get) == "a" and sum(star.values()) == pytest.approx(1.0)


def test_hits_separates_hubs_and_authorities():
    nodes = ["h1", "h2", "x", "y"]
    hub, auth = hits(nodes, [("h1", "x"), ("h1", "y"), ("h2", "x")])
    assert auth["x"] > auth["y"] > auth["h1"] == auth["h2"] == 0
    assert hub["h1"] > hub["h2"] > hub["x"] == 0
    assert sum(v * v for v in hub.values()) == pytest.approx(1.0) and sum(v * v for v in auth.values()) == pytest.approx(1.0)


COMPANIES = [{"ticker": "AAPL", "name": "Apple Inc.", "aliases": ["Apple"]},
             {"ticker": "MSFT", "name": "Microsoft Corporation", "aliases": ["Microsoft"]},
             {"ticker": "BA", "name": "The Boeing Company", "aliases": ["Boeing"]}]


def P(i, ticker, text):
    return Passage(id=i, ticker=ticker, company=ticker, fiscal_year=2025, item="1A", kind="text", text=text)


def test_mentions_are_strict_and_exclude_self():
    ps = [P("1", "AAPL", "We compete with Microsoft in some markets."), P("2", "AAPL", "Microsoft Corporation supplies software."),
          P("3", "AAPL", "Apple depends on suppliers."),                 # self mention: ignored
          P("4", "AAPL", "microsoft in lower case is not matched"),
          P("5", "MSFT", "Apple sells devices; BA is a ticker we do not match."),
          P("6", "MSFT", "Pineapple growers are not Apple."),           # 'Pineapple' must not match, 'Apple.' does
          P("7", "AAPL", "Boeing aircraft")]
    counts, evidence = count_mentions(ps, COMPANIES)
    assert counts == {("AAPL", "MSFT"): 2, ("AAPL", "BA"): 1, ("MSFT", "AAPL"): 2}
    g = build_graph(ps, COMPANIES, m=2)
    assert {(e["source"], e["target"]) for e in g["edges"]} == {("AAPL", "MSFT"), ("MSFT", "AAPL")}
    assert sum(n["pagerank"] for n in g["nodes"]) == pytest.approx(1.0)
    assert all({"ticker", "pagerank", "hub", "authority", "in_degree"} <= set(n) for n in g["nodes"])
    json.dumps(g)


def test_graph_matches_the_apps_validator(tmp_path):
    from trademesh.app.backend import load_graph
    ps = [P("1", "AAPL", "Microsoft Microsoft"), P("2", "AAPL", "Microsoft again")]
    f = tmp_path / "g.json"
    f.write_text(json.dumps(build_graph(ps, COMPANIES)))
    data, msg = load_graph(f)
    assert data is not None and msg == ""


def test_maxsim_by_hand():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    page = np.array([[0.9, 0.1], [0.2, 0.8], [0.0, 0.0]])
    # token 1 best match 0.9, token 2 best match 0.8
    assert maxsim(q, page) == pytest.approx(1.7)
    assert maxsim(np.zeros((0, 2)), page) == 0.0
    pages = {"A_1": page, "A_2": np.array([[0.1, 0.1]]), "A_3": np.array([[1.0, 1.0]])}
    assert [p for p, _ in rank_pages(q, pages, 2)] == ["A_3", "A_1"]            # A_3: 1 + 1 = 2.0 > 1.7


def test_vision_results_file_matches_the_apps_validator(tmp_path):
    from trademesh.app.backend import load_vision
    write_results(tmp_path / "r.json", [{"query": "q", "results": [{"ticker": "AAPL", "page": 4, "score": 1.5, "image": "x.png"}]}])
    assert load_vision(tmp_path / "r.json")[0] is not None


def test_vision_scoring_ranks_the_matching_page_first_and_the_app_reads_the_extra_fields(tmp_path, monkeypatch):
    """eval/vision_score.py on tiny made-up embeddings and page texts: MaxSim finds the page whose vectors match the query."""
    import json
    import sys
    from pathlib import Path

    import numpy as np

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import eval.vision_score as vs

    v = tmp_path / "vision"
    v.mkdir()
    pages = {"A_p1": {"ticker": "A", "page": 1, "image": "a.png", "text": "revenue by segment cloud", "kind": "gold"},
             "A_p2": {"ticker": "A", "page": 2, "image": "b.png", "text": "risk factors competition", "kind": "prose"}}
    questions = [{"id": "Q1", "question": "revenue by segment", "company": "A", "gold": ["A_p1"]}]
    (v / "pages.json").write_text(json.dumps(pages))
    (v / "questions.json").write_text(json.dumps(questions))
    np.savez(v / "emb_pages.npz", A_p1=np.array([[1.0, 0.0], [0.9, 0.1]]), A_p2=np.array([[0.0, 1.0]]))
    np.savez(v / "emb_queries.npz", Q1=np.array([[1.0, 0.0]]))
    monkeypatch.setattr(vs, "V", v)
    monkeypatch.setattr(vs, "ROOT", tmp_path)
    vs.main()
    res = json.loads((v / "results.json").read_text())
    assert res["queries"][0]["results"][0]["page"] == 1 and res["queries"][0]["results"][0]["correct"] is True
    assert res["summary"]["vision_maxsim"]["hit@1"] == 1 and res["summary"]["text_bm25"]["hit@1"] == 1
    from trademesh.app.backend import load_vision
    assert load_vision(v / "results.json")[0] is not None


def test_full_scoring_questions_and_app_loader(tmp_path, monkeypatch):
    """vision_full_prepare builds company questions from page texts; vision_full_score ranks pages; the app prefers the full results."""
    import json
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import eval.vision_full_prepare as fp
    import eval.vision_full_score as fs

    pages = {"A_p1": {"ticker": "A", "page": 1, "image": "pages/A_p1.png", "digits": 0.3, "text": "Consolidated Balance Sheets Total assets 10 Total liabilities 5"},
             "A_p2": {"ticker": "A", "page": 2, "image": "pages/A_p2.png", "digits": 0.02, "text": "Balance Sheets are discussed in prose Total assets Total liabilities"},
             "B_p1": {"ticker": "B", "page": 1, "image": "pages/B_p1.png", "digits": 0.3, "text": "Statements of Cash Flows operating activities investing financing 5"}}
    qs, dropped = fp.build_questions(pages, {"A": "Alpha", "B": "Beta"})
    by = {q["id"]: q for q in qs}
    assert by["A-balance"]["gold"] == ["A_p1"] and by["A-balance"]["question"].startswith("Alpha ")      # the prose page is not gold
    assert by["B-cash"]["gold"] == ["B_p1"] and any(d.startswith("A-income") for d in dropped)

    work = tmp_path / "work"
    (work / "emb").mkdir(parents=True)
    (work / "pages").mkdir()
    for pid in pages:
        Image.new("RGB", (20, 30), "white").save(work / "pages" / f"{pid}.png")
    (work / "pages.json").write_text(json.dumps(pages))
    keep = [q for q in qs if q["id"] in ("A-balance", "B-cash")]
    (work / "questions.json").write_text(json.dumps(keep))
    vecs = {"A_p1": [[1.0, 0.0]], "A_p2": [[0.0, 1.0]], "B_p1": [[0.6, 0.6]]}
    for pid, v in vecs.items():
        np.save(work / "emb" / f"{pid}.npy", np.array(v, dtype=np.float16))
    np.savez(work / "emb_queries.npz", **{"A-balance": np.array([[1.0, 0.0]]), "B-cash": np.array([[0.6, 0.6]])})
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--work", str(work), "--out", str(out)])
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "companies.json").write_text(json.dumps([{"ticker": "A", "name": "Alpha"}, {"ticker": "B", "name": "Beta"}]))
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    fs.main()
    res = json.loads((out / "results.json").read_text())
    assert res["summary"]["overall"]["vision_maxsim"]["hit@1"] == 2 and res["scale"] == "full"
    assert (out / "thumbs" / "A_p1.jpg").exists() and res["summary"]["groups"]["balance sheet"]["vision_maxsim"]["n"] == 1

    from trademesh.app import backend
    monkeypatch.setattr(backend, "VISION_FULL_PATH", out / "results.json")
    monkeypatch.setattr(backend, "VISION_PATH", tmp_path / "missing.json")
    assert backend.load_vision()[0]["scale"] == "full"
