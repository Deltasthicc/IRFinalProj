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
