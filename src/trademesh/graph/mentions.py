"""Company mention graph with PageRank and HITS (Role A, proposal 9.7).

Edge A -> B if at least m passages of A's filing mention company B ("A mentions B"). Mentions are found
by STRICT matching: whole-word, case-sensitive company names and aliases (not tickers, which collide
with ordinary words such as BA, KO or DIS), never the company's own names. This is deliberately stricter
than the query-time entity matcher, which tolerates typos: in running prose a fuzzy match would
create false edges.

    PageRank   PR(v) = (1 - d)/|V| + d * sum_{u -> v} PR(u)/outdeg(u),  d = 0.85, power iteration;
               dangling companies (no outgoing edge) spread their rank uniformly.
    HITS       a = A^T h,  h = A a,  both L2-normalised every step (authority a, hub h).

There is no ground truth for "importance": we report the rankings and how they correlate with simple
in-degree, and make no business claims.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from ..evaluation.metrics import kendall_tau, spearman
from ..types import Passage

LEGAL_SUFFIX = re.compile(r",?\s+(Inc|Corp|Corporation|Co|Company|Ltd|plc|Incorporated)\.?$")


def name_variants(company: dict) -> list[str]:
    names = [company["name"], LEGAL_SUFFIX.sub("", company["name"])] + list(company.get("aliases", []))
    names = [n.strip() for n in names if len(n.strip()) >= 4]
    return sorted(set(names), key=len, reverse=True)


def build_patterns(companies: list[dict]) -> dict[str, re.Pattern]:
    pats = {}
    for c in companies:
        alts = "|".join(re.escape(n) for n in name_variants(c))
        pats[c["ticker"]] = re.compile(rf"(?<![A-Za-z0-9])(?:{alts})(?![A-Za-z0-9])")
    return pats


def count_mentions(passages: list[Passage], companies: list[dict]) -> tuple[dict, dict]:
    """Return ({(A, B): n passages of A mentioning B}, {(A, B): [first passage ids]})."""
    pats = build_patterns(companies)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    evidence: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in passages:
        for b, pat in pats.items():
            if b != p.ticker and pat.search(p.text):
                counts[(p.ticker, b)] += 1
                if len(evidence[(p.ticker, b)]) < 3:
                    evidence[(p.ticker, b)].append(p.id)
    return dict(counts), dict(evidence)


def pagerank(nodes: list[str], edges: list[tuple[str, str]], d: float = 0.85, tol: float = 1e-12, max_iter: int = 1000) -> dict[str, float]:
    n = len(nodes)
    out = defaultdict(list)
    for u, v in edges:
        out[u].append(v)
    pr = {v: 1 / n for v in nodes}
    for _ in range(max_iter):
        dangling = sum(pr[v] for v in nodes if not out[v])
        new = {v: (1 - d) / n + d * dangling / n for v in nodes}
        for u in nodes:
            for v in out[u]:
                new[v] += d * pr[u] / len(out[u])
        if sum(abs(new[v] - pr[v]) for v in nodes) < tol:
            pr = new
            break
        pr = new
    return pr


def hits(nodes: list[str], edges: list[tuple[str, str]], tol: float = 1e-12, max_iter: int = 1000) -> tuple[dict, dict]:
    """Return (hub, authority), each L2-normalised."""
    hub = {v: 1.0 for v in nodes}
    auth = {v: 1.0 for v in nodes}
    for _ in range(max_iter):
        new_a = {v: 0.0 for v in nodes}
        for u, v in edges:
            new_a[v] += hub[u]
        na = math.sqrt(sum(x * x for x in new_a.values())) or 1.0
        new_a = {v: x / na for v, x in new_a.items()}
        new_h = {v: 0.0 for v in nodes}
        for u, v in edges:
            new_h[u] += new_a[v]
        nh = math.sqrt(sum(x * x for x in new_h.values())) or 1.0
        new_h = {v: x / nh for v, x in new_h.items()}
        done = sum(abs(new_a[v] - auth[v]) + abs(new_h[v] - hub[v]) for v in nodes) < tol
        hub, auth = new_h, new_a
        if done:
            break
    return hub, auth


def build_graph(passages: list[Passage], companies: list[dict], m: int = 2) -> dict:
    counts, evidence = count_mentions(passages, companies)
    tickers = [c["ticker"] for c in companies]
    edges = [(a, b) for (a, b), n in sorted(counts.items()) if n >= m]
    pr = pagerank(tickers, edges)
    hub, auth = hits(tickers, edges)
    indeg = {t: sum(1 for _, v in edges if v == t) for t in tickers}
    outdeg = {t: sum(1 for u, _ in edges if u == t) for t in tickers}
    names = {c["ticker"]: c["name"] for c in companies}
    rank = lambda d: [d[t] for t in tickers]
    corr = {}
    if len(tickers) >= 3:
        corr = {"pagerank_vs_indegree_spearman": spearman(rank(pr), rank(indeg)),
                "authority_vs_indegree_spearman": spearman(rank(auth), rank(indeg)),
                "pagerank_vs_authority_spearman": spearman(rank(pr), rank(auth)),
                "pagerank_vs_indegree_kendall": kendall_tau(rank(pr), rank(indeg))}
    return {"params": {"min_passages_per_edge": m, "damping": 0.85, "n_edges": len(edges)},
            "nodes": [{"ticker": t, "name": names[t], "pagerank": pr[t], "hub": hub[t], "authority": auth[t],
                       "in_degree": indeg[t], "out_degree": outdeg[t]} for t in tickers],
            "edges": [{"source": a, "target": b, "weight": counts[(a, b)], "evidence": evidence[(a, b)]} for a, b in edges],
            "correlations": corr}


def main(argv=None) -> int:
    import argparse
    from ..data import DATA, load_corpus
    ap = argparse.ArgumentParser(description="Build data/graph/company_graph.json from the corpus.")
    ap.add_argument("--min-passages", type=int, default=2)
    ap.add_argument("--allow-fixture", action="store_true")
    args = ap.parse_args(argv)
    passages, companies, is_fixture = load_corpus()
    if is_fixture and not args.allow_fixture:
        raise SystemExit("Refusing to build the graph from the test fixture (invented text). Build the corpus first.")
    g = build_graph(passages, companies, args.min_passages)
    out = DATA / "graph" / ("company_graph.json" if not is_fixture else "company_graph_fixture.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g, indent=1))
    top = sorted(g["nodes"], key=lambda n: -n["pagerank"])[:5]
    print(f"{len(g['edges'])} edges; top PageRank: " + ", ".join(f"{n['ticker']} {n['pagerank']:.3f}" for n in top))
    print("correlations:", json.dumps(g["correlations"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
