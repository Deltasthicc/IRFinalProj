"""Extra figures for the short report, all drawn from files in the repository.

    python docs/report/figures/make_report_extras.py
Writes fig_corpus.png (what is in the corpus), fig_glance.png (four headline results) and fig_graph_net_s.png (company graph).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
DARK, MID, LIGHT, ORANGE = "#1f3a5f", "#4f81bd", "#9dc3e6", "#d9822b"


def J(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def corpus():
    per = Counter()
    tab = Counter()
    items = Counter()
    for line in (ROOT / "data/passages.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        per[d["ticker"]] += 1
        tab[d["ticker"]] += d.get("kind") == "table"
        items[d["item"]] += 1
    order = [t for t, _ in per.most_common()]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    text = [per[t] - tab[t] for t in order]
    a.bar(order, text, color=MID, label="text passages")
    a.bar(order, [tab[t] for t in order], bottom=text, color=ORANGE, label="table chunks")
    a.set_ylabel("passages")
    a.set_title("Passages per company (11,964 in total)")
    a.tick_params(axis="x", rotation=60, labelsize=6.5)
    a.legend(frameon=False, fontsize=7)
    top = items.most_common(8)
    names = [("other" if k == "X" else f"Item {k}") for k, _ in top]
    b.barh(names[::-1], [v for _, v in top][::-1], color=DARK)
    b.set_title("Passages per 10-K section (top 8)")
    b.set_xlabel("passages")
    fig.tight_layout()
    fig.savefig(HERE / "fig_corpus.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def glance():
    ret, hum = J("eval/out/retrieval_test.json")["overall"], J("eval/out/rag_scores_human.json")
    ent, ab = J("eval/out/entity_eval.json"), J("eval/out/abstention_eval.json")["test"]
    tiles = [(f"{ret['hybrid+rerank']['nDCG@10']:.3f}", "nDCG@10, best search", f"BM25: {ret['bm25']['nDCG@10']:.3f}", DARK),
             (f"{hum['configs']['best_rag']['mean_rating']:.2f} / 2", "answer rating (0-2)", f"no retrieval: {hum['configs']['closed_book']['mean_rating']:.3f}", MID),
             (f"{ent['accuracy'] * 100:.1f}%", "names fixed", f"{ent['correct']} of {ent['n']}", ORANGE),
             (f"{ab['tp']} / {ab['tp'] + ab['fn']}", "refusals correct", f"of {ab['tp'] + ab['fn']} unanswerable", "#2f7d4f")]
    fig, ax = plt.subplots(figsize=(6.5, 0.95))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.axis("off")
    for i, (big, label, sub, col) in enumerate(tiles):
        x = 1 + i * 25
        ax.add_patch(FancyBboxPatch((x, 1), 22.5, 18, boxstyle="round,pad=0.3,rounding_size=1.5", fc="white", ec=col, lw=1.2))
        ax.text(x + 11.25, 12.5, big, ha="center", va="center", fontsize=14, fontweight="bold", color=col)
        ax.text(x + 11.25, 6.8, label, ha="center", va="center", fontsize=6.2)
        ax.text(x + 11.25, 3.5, sub, ha="center", va="center", fontsize=5.8, color="#555")
    fig.savefig(HERE / "fig_glance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def network():
    g = J("data/graph/company_graph.json")
    G = nx.DiGraph()
    pr = {n["ticker"]: n["pagerank"] for n in g["nodes"]}
    G.add_nodes_from(pr)
    for e in g["edges"]:
        G.add_edge(e["source"], e["target"], w=e.get("weight", 1))
    pos = nx.spring_layout(G, seed=7, k=0.9)
    fig, ax = plt.subplots(figsize=(3.35, 2.9))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888", arrowsize=8, width=[min(0.5 + 0.15 * G.edges[e]["w"], 2.5) for e in G.edges])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[4000 * pr[n] + 60 for n in G], node_color=LIGHT, edgecolors=DARK, linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
    ax.set_title("Who mentions whom (node size = PageRank)", fontsize=8)
    ax.axis("off")
    fig.savefig(HERE / "fig_graph_net_s.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def ablation():
    """Forest plot of the one-change-at-a-time table: difference in nDCG@10 from plain BM25, with 95% intervals."""
    a = J("eval/out/ablation_test.json")
    pick = [("A prep:stop+nostem+num+meta", "company/year/section added"),
            ("A prep:stop+nostem+nonum+meta", "same, numbers not indexed"),
            ("A prep:stop+stem+num+nometa", "stemming"),
            ("B struct:drop table chunks", "no table chunks"),
            ("B struct:entity filter off", "no company filter"),
            ("D prf:k=5, m=10", "feedback (PRF)"),
            ("C fusion:rrf k=60", "hybrid, k = 60"),
            ("C fusion:rrf k=10", "hybrid, k = 10")]
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    for i, (k, label) in enumerate(pick[::-1]):
        d = a[k]["vs_default"]
        clear = d["ci_low"] > 0 or d["ci_high"] < 0
        col = DARK if clear else "#8a8a8a"
        ax.plot([d["ci_low"], d["ci_high"]], [i, i], color=col, lw=1.4)
        ax.plot(d["mean_diff"], i, "o", color=col, ms=4)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(range(len(pick)))
    ax.set_yticklabels([lbl for _, lbl in pick[::-1]], fontsize=6.5)
    ax.set_xlabel("change in nDCG@10 vs plain BM25 (95% interval)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title("One change at a time (dark = interval excludes 0)", fontsize=7.5)
    ax.grid(axis="x", color="#ddd", lw=0.4)
    fig.savefig(HERE / "fig_ablation_s.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def hypotheses():
    """Scoreboard of the 11 hypotheses (status words follow the decision rules; see the hypotheses table in the paper)."""
    rows = [("H1", "BM25 beats TF-IDF", "supported"), ("H2", "BM25 vs dense by question type", "unclear"),
            ("H3", "hybrid beats BM25 and dense", "partly"), ("H4", "reranker improves hybrid", "unclear"),
            ("H5", "table chunks and numbers help", "unclear"), ("H6", "feedback helps", "not supported"),
            ("H7", "champion lists faster, same quality", "partly"), ("H8", "answers with retrieval win", "supported"),
            ("H9", "best retriever beats BM25 answers", "supported"), ("H10", "PageRank differs from counting", "not supported"),
            ("H11", "page images beat text", "not tested")]
    col = {"supported": "#2f7d4f", "partly": "#8fbf6a", "unclear": "#e0a030", "not supported": "#c0504d", "not tested": "#9a9a9a"}
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.axis("off")
    for i, (h, text, st) in enumerate(rows):
        ax.text(0.0, i, h, va="center", fontsize=6.8, fontweight="bold")
        ax.text(0.9, i, text, va="center", fontsize=6.6)
        ax.add_patch(FancyBboxPatch((7.2, i - 0.32), 2.7, 0.64, boxstyle="round,pad=0.02,rounding_size=0.25", fc=col[st], ec="none"))
        ax.text(8.55, i, st, va="center", ha="center", fontsize=6, color="white", fontweight="bold")
    ax.set_title("Hypotheses at a glance", fontsize=7.5)
    fig.savefig(HERE / "fig_hypotheses_s.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def rating_distribution():
    """How often each rater gave 0, 1 or 2 to the answers of each configuration (from the rater sheets)."""
    import csv
    key = J("eval/human/key_DO_NOT_OPEN.json")
    cfg = [("closed_book", "no retrieval"), ("bm25_rag", "BM25 + answer"), ("best_rag", "hybrid + rerank")]
    raters = [("Shashwat", "Shashwat"), ("Ruchir", "Ruchir")]
    counts = {}
    for folder, _ in raters:
        for r in csv.DictReader(open(ROOT / "eval/human/raters" / folder / "ratings.csv", encoding="utf-8")):
            c = key[r["question_id"]][r["label"]]
            counts.setdefault((folder, c), Counter())[int(r["rating"])] += 1
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    labels, y = [], 0
    colors = {0: "#c0504d", 1: "#9dc3e6", 2: DARK}
    for c, cname in cfg[::-1]:
        for folder, rname in raters[::-1]:
            left = 0
            for score in (0, 1, 2):
                v = counts[(folder, c)][score]
                ax.barh(y, v, left=left, color=colors[score], edgecolor="white", lw=0.5, label=f"rated {score}" if y == 0 else None)
                if v:
                    ax.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=6, color="white" if score != 1 else "black")
                left += v
            labels.append(f"{cname}, {rname}")
            y += 1
        y += 0.4
    ys = []
    y = 0
    for _ in cfg:
        for _ in raters:
            ys.append(y)
            y += 1
        y += 0.4
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("number of the 20 answers", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title("What each rater gave (0, 1 or 2)", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.28))
    fig.savefig(HERE / "fig_rating_dist_s.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    corpus()
    glance()
    network()
    ablation()
    hypotheses()
    rating_distribution()
    print("wrote fig_corpus.png, fig_glance.png, fig_graph_net_s.png, fig_ablation_s.png")
