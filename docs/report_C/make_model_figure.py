"""Generates docs/report_C/fig_model.svg (Figure 1 of the report: the TradeMesh model).

Run:  python docs/report_C/make_model_figure.py
Pure Python, no plotting library. Boxes outlined in orange are the query / RAG / app components.
"""
from html import escape
from pathlib import Path

W, H = 1000, 560
BOXES, ARROWS = [], []
COL = {"off": ("#e3f2ec", "#1b7f5c"), "on1": ("#e1ebfa", "#2d62b5"), "on2": ("#ece6fa", "#5b3fb0"),
       "ev": ("#eeeeee", "#666666"), "gate": ("#f0eafd", "#7a5fc0")}


def box(x, y, w, h, title, sub, kind, mine=False, dashed=False):
    fill, stroke = COL[kind]
    extra = ' stroke-dasharray="6 4"' if dashed else ""
    sw = 3 if mine else 1.5
    stroke = "#d9822b" if mine else stroke
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'
    s += f'<text x="{x + w / 2}" y="{y + 24}" text-anchor="middle" font-size="13" font-weight="700">{escape(title)}</text>'
    for i, line in enumerate(sub):
        s += f'<text x="{x + w / 2}" y="{y + 42 + 14 * i}" text-anchor="middle" font-size="10.5" fill="#444">{escape(line)}</text>'
    BOXES.append(s)


def arrow(points):
    d = "M " + " L ".join(f"{x},{y}" for x, y in points)
    ARROWS.append(f'<path d="{d}" fill="none" stroke="#555" stroke-width="1.6" marker-end="url(#a)"/>')


def label(x, y, text, color):
    BOXES.append(f'<text x="{x}" y="{y}" font-size="12" font-weight="700" fill="{color}">{escape(text)}</text>')


X = [20, 260, 500, 740]
BW, BH = 200, 72

label(20, 26, "OFFLINE: build the collection, indexes and graph (Role A)", "#1b7f5c")
box(X[0], 36, BW, BH, "SEC EDGAR", ["20 companies, latest 10-K"], "off")
box(X[1], 36, BW, BH, "Polite crawler", ["User-Agent, rate limit, cache"], "off")
box(X[2], 36, BW, BH, "Parse + chunk", ["Items, passages, one chunk", "per table"], "off")
box(X[3], 36, BW, BH, "Text indexes", ["positional index, k-gram", "dense vectors"], "off")
box(X[2], 138, BW, 60, "Vision track (gated)", ["page images, MaxSim"], "gate", dashed=True)
box(X[3], 138, BW, 60, "Company mention graph", ["PageRank + HITS"], "off")
arrow([(X[0] + BW, 72), (X[1], 72)]); arrow([(X[1] + BW, 72), (X[2], 72)]); arrow([(X[2] + BW, 72), (X[3], 72)])
arrow([(X[2] + 60, 108), (X[2] + 60, 138)])
arrow([(X[2] + 140, 108), (X[2] + 140, 122), (X[3] + 100, 122), (X[3] + 100, 138)])

label(20, 232, "ONLINE 1: retrieve (query side is Role C, rankers are Role B)", "#2d62b5")
box(X[0], 242, BW, BH, "User question", ['e.g. "Compare Appel and', 'Tesla supply risk"'], "on1", mine=True)
box(X[1], 242, BW, BH, "Query processor", ["entity + spelling (bigram, Levenshtein)", "compare split, section prior"], "on1", mine=True)
box(X[2], 242, BW, BH, "Retrievers", ["Boolean, TF-IDF, BM25, dense,", "hybrid (RRF), champion lists"], "on1")
box(X[3], 242, BW, BH, "Cross-encoder reranker", ["re-scores pairs, keeps top-5"], "on1")
arrow([(X[0] + BW, 278), (X[1], 278)]); arrow([(X[1] + BW, 278), (X[2], 278)]); arrow([(X[2] + BW, 278), (X[3], 278)])

label(20, 372, "ONLINE 2: generate a cited, checked answer (Role C, flows right to left)", "#5b3fb0")
box(X[3], 382, BW, BH, "Context builder", ["top-5 passages labelled", "[P1]..[P5] + metadata"], "on2", mine=True)
box(X[2], 382, BW, BH, "LLM (temp 0, cached)", ["answer only from passages,", "cite, or INSUFFICIENT EVIDENCE"], "on2", mine=True)
box(X[1], 382, BW, BH, "Verifier + abstention", ["cited ids exist, numbers in", "cited text, score vs tau"], "on2", mine=True)
box(X[0], 382, BW, BH, "Streamlit app", ["Ask | Search | Filing graph | Pages", "trust panel"], "on2", mine=True)
arrow([(X[3] + 100, 314), (X[3] + 100, 382)])
arrow([(X[3], 418), (X[2] + BW, 418)]); arrow([(X[2], 418), (X[1] + BW, 418)]); arrow([(X[1], 418), (X[0] + BW, 418)])

label(20, 492, "EVALUATION (separate scripts, same pipeline)", "#666666")
box(X[0], 502, BW, 50, "Test collection", ["30 needs, pooled judgments"], "ev")
box(X[1], 502, BW, 50, "Retrieval metrics", ["P/R/F1, MAP, nDCG, MRR"], "ev")
box(X[2], 502, BW, 50, "RAG + abstention (C)", ["ratings, citations, P/R/F1"], "ev", mine=True)
box(X[3], 502, BW, 50, "Graph + vision", ["PageRank vs in-degree"], "ev")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">'
       '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
       '<path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="white"/>' + "".join(ARROWS) + "".join(BOXES) +
       '<rect x="720" y="470" width="14" height="14" rx="3" fill="white" stroke="#d9822b" stroke-width="3"/>'
       '<text x="740" y="481" font-size="11" fill="#444">orange outline: query, RAG and app (Role C)</text></svg>')
out = Path(__file__).with_name("fig_model.svg")
out.write_text(svg, encoding="utf-8")
print(f"wrote {out}")
