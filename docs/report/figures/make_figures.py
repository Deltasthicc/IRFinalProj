"""Draw the report result charts from the stored result files.

Run from the repo root:
    .\\.venv\\Scripts\\python.exe docs/report/figures/make_figures.py

Every plotted number is read from JSON; nothing is typed by hand.
Output: PNG files (200 dpi) next to this script. Deterministic.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

W = 6.5  # inches
DPI = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "hatch.linewidth": 0.8,
    "svg.hashsalt": "tradmesh",
})

# Colours chosen to differ in lightness (grayscale-safe); hatches add a second cue.
C_DARK, C_MID, C_LIGHT, C_PALE = "#1f3a5f", "#4f81bd", "#9dc3e6", "#e0e0e0"
C_ORANGE = "#d9822b"
OUTPUTS = []


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08,
                metadata={"Software": None})
    plt.close(fig)
    OUTPUTS.append(path)


LABELS = {
    "tfidf": "TF-IDF",
    "tfidf-champ-50": "TF-IDF, champion list 50",
    "tfidf-champ-200": "TF-IDF, champion list 200",
    "boolean": "Boolean",
    "bm25": "BM25",
    "bm25+prf": "BM25 + PRF",
    "dense": "Dense",
    "hybrid": "Hybrid (RRF)",
    "bm25+rerank": "BM25 + rerank",
    "hybrid+rerank": "Hybrid + rerank",
}
# system -> (colour, hatch, marker, linestyle)
STYLE = {
    "tfidf": (C_PALE, "", "s", ":"),
    "tfidf-champ-50": (C_PALE, "..", "s", ":"),
    "tfidf-champ-200": (C_PALE, "//", "s", ":"),
    "boolean": (C_PALE, "xx", "x", ":"),
    "bm25": (C_LIGHT, "", "o", "-"),
    "bm25+prf": (C_LIGHT, "//", "o", "--"),
    "dense": (C_MID, "", "^", "-."),
    "hybrid": (C_MID, "\\\\", "D", "-"),
    "bm25+rerank": (C_DARK, "//", "v", "--"),
    "hybrid+rerank": (C_DARK, "", "*", "-"),
}


def style_bar(bars, colour, hatch):
    for b in bars:
        b.set_facecolor(colour)
        b.set_hatch(hatch)
        b.set_edgecolor("black")
        b.set_linewidth(0.8)


# --------------------------------------------------------------------- fig 1
def fig_ndcg_systems():
    r = load("eval/out/retrieval_test.json")
    metric = "nDCG@10"
    overall = r["overall"]
    boot = r.get("bootstrap", {})
    base = r.get("baseline", "bm25")
    systems = sorted(overall, key=lambda s: (-overall[s][metric], s))
    vals = [overall[s][metric] for s in systems]

    fig = plt.figure(figsize=(W, 3.6))
    ax = fig.add_axes([0.30, 0.15, 0.30, 0.76])
    y = list(range(len(systems)))
    bars = ax.barh(y, vals, height=0.7)
    for b, s in zip(bars, systems):
        style_bar([b], STYLE[s][0], STYLE[s][1])
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[s] for s in systems])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"{metric} (mean, {r['n_needs']} test questions)")
    ax.set_title(f"{metric} by retrieval system (test set)", loc="left", x=-0.95)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", ha="left")
    # The stored bootstrap entries are paired differences vs the baseline,
    # not per-system intervals, so they are printed as text, not error bars.
    ax.text(1.12, -0.9, f"Diff. vs {LABELS[base]}\n[95% CI]", ha="left",
            va="bottom", fontsize=9, fontweight="bold", clip_on=False)
    for yi, s in zip(y, systems):
        if s in boot and metric in boot[s]:
            d = boot[s][metric]
            txt = f"{d['mean_diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
        elif s == base:
            txt = "(reference)"
        else:
            txt = ""
        ax.text(1.12, yi, txt, va="center", ha="left", fontsize=9, clip_on=False)
    ax.grid(axis="x", color="#cccccc", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(HERE, "fig_ndcg_systems.png"), dpi=DPI,
                metadata={"Software": None})
    OUTPUTS.append(os.path.join(HERE, "fig_ndcg_systems.png"))
    plt.close(fig)


# --------------------------------------------------------------------- fig 2
def fig_pr_curves():
    r = load("eval/out/retrieval_test.json")
    pr = r["pr_11pt"]
    systems = [s for s in ["tfidf", "bm25", "dense", "hybrid", "hybrid+rerank"]
               if s in pr]
    colours = {"tfidf": "#7f7f7f", "bm25": C_ORANGE, "dense": C_MID,
               "hybrid": "#2e8b57", "hybrid+rerank": "black"}
    fig, ax = plt.subplots(figsize=(W, 3.8))
    for s in systems:
        yv = pr[s]
        xv = [i / (len(yv) - 1) for i in range(len(yv))]
        ax.plot(xv, yv, color=colours[s], marker=STYLE[s][2],
                linestyle=STYLE[s][3], linewidth=1.5, markersize=5.5,
                label=LABELS[s])
    ax.set_xlabel("Recall (fraction)")
    ax.set_ylabel("Interpolated precision (fraction)")
    ax.set_title("11-point interpolated precision-recall (test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#cccccc", linewidth=0.5)
    ax.legend(loc="lower left", frameon=True)
    save(fig, "fig_pr_curves.png")


# --------------------------------------------------------------------- fig 3
def fig_ndcg_by_type():
    r = load("eval/out/retrieval_test.json")
    bt = r["by_type"]
    systems = [s for s in ["bm25", "hybrid", "hybrid+rerank"] if s in bt]
    types = list(bt[systems[0]].keys())
    fig, ax = plt.subplots(figsize=(W, 3.6))
    n = len(systems)
    width = 0.8 / n
    for i, s in enumerate(systems):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(types))]
        vals = [bt[s][t]["nDCG@10"] for t in types]
        bars = ax.bar(xs, vals, width * 0.95, label=LABELS[s])
        style_bar(bars, STYLE[s][0], STYLE[s][1])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_xlabel("Information-need type")
    ax.set_ylabel("nDCG@10 (mean per type)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"nDCG@10 by need type (test, {r['n_needs']} questions)")
    ax.legend(loc="upper center", ncol=n, frameon=False,
              bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="y", color="#cccccc", linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig_ndcg_by_type.png")


# --------------------------------------------------------------------- fig 4
def fig_rag_ratings():
    h = load("eval/out/rag_scores_human.json")
    m = load("eval/out/rag_scores_test.json")
    cfgs = ["closed_book", "bm25_rag", "best_rag"]
    names = {"closed_book": "Closed-book", "bm25_rag": "BM25-RAG",
             "best_rag": "Best-RAG"}
    hv = [h["configs"][c]["mean_rating"] for c in cfgs]
    mv = [m["configs"][c]["mean_rating"] for c in cfgs]
    nq = h["n_questions"]

    fig, ax = plt.subplots(figsize=(W, 4.0))
    width = 0.36
    xs = list(range(len(cfgs)))
    b1 = ax.bar([x - width / 2 for x in xs], hv, width, label="Human rating")
    style_bar(b1, C_DARK, "")
    b2 = ax.bar([x + width / 2 for x in xs], mv, width,
                label="Model-assisted rating")
    style_bar(b2, C_LIGHT, "//")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                    f"{b.get_height():.2f}", ha="center", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([names[c] for c in cfgs])
    ax.set_ylabel("Mean answer rating (scale 0-2)")
    ax.set_xlabel("Answer configuration")
    ax.set_ylim(0, 2.75)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_title(f"Mean answer rating ({nq} test questions)")
    ax.legend(loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="y", color="#cccccc", linewidth=0.5)
    ax.set_axisbelow(True)
    lines = ["Human paired differences (95% bootstrap CI):"]
    pretty = {"bm25_rag": "BM25-RAG", "closed_book": "closed-book",
              "best_rag": "best-RAG"}
    for k, d in h["bootstrap"].items():
        a, b = [t.strip() for t in k.split(" - ")]
        lines.append(f"{pretty[a]} - {pretty[b]}: {d['mean_diff']:+.2f} "
                     f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]")
    ax.text(0.99, 0.99, "\n".join(lines), transform=ax.transAxes, ha="right",
            va="top", fontsize=9,
            bbox=dict(boxstyle="square", facecolor="white", edgecolor="black",
                      linewidth=0.8))
    save(fig, "fig_rag_ratings.png")


# --------------------------------------------------------------------- fig 5
def fig_entity_accuracy():
    e = load("eval/out/entity_eval.json")
    bk = e["by_kind"]
    kinds = sorted(bk, key=lambda k: (-bk[k]["n"], k))
    acc = [100.0 * bk[k]["correct"] / bk[k]["n"] for k in kinds]
    fig, ax = plt.subplots(figsize=(W, 3.6))
    bars = ax.bar(range(len(kinds)), acc)
    for i, b in enumerate(bars):
        full = bk[kinds[i]]["correct"] == bk[kinds[i]]["n"]
        b.set_facecolor(C_MID if full else C_ORANGE)
        b.set_hatch("" if full else "//")
        b.set_edgecolor("black")
        b.set_linewidth(0.8)
    for i, k in enumerate(kinds):
        ax.text(i, acc[i] + 1.5, f"{bk[k]['correct']}/{bk[k]['n']}\n({acc[i]:.0f}%)",
                ha="center", va="bottom")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([k.capitalize() for k in kinds])
    ax.set_xlabel("Typo kind (correct/total above each bar)")
    ax.set_ylabel("Entity accuracy (%)")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title(f"Entity resolution accuracy: {e['correct']}/{e['n']} "
                 f"({100 * e['accuracy']:.1f}%)")
    ax.grid(axis="y", color="#cccccc", linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig_entity_accuracy.png")


# --------------------------------------------------------------------- fig 6
def fig_pagerank():
    g = load("data/graph/company_graph.json")
    nodes = sorted(g["nodes"], key=lambda n: (-n["pagerank"], n["ticker"]))[:12]
    cor = g["correlations"]
    fig, ax = plt.subplots(figsize=(W, 4.0))
    xs = list(range(len(nodes)))
    bars = ax.bar(xs, [n["pagerank"] for n in nodes], 0.65)
    style_bar(bars, C_LIGHT, "//")
    ax.set_xticks(xs)
    ax.set_xticklabels([n["ticker"] for n in nodes], rotation=45, ha="right")
    ax.set_xlabel("Company (ticker), top 12 by PageRank")
    ax.set_ylabel("PageRank score (sums to 1)")
    ax.set_ylim(0, max(n["pagerank"] for n in nodes) * 1.2)
    ax.grid(axis="y", color="#cccccc", linewidth=0.5)
    ax.set_axisbelow(True)
    for x, n in zip(xs, nodes):
        ax.text(x, n["pagerank"] + 0.002, f"{n['pagerank']:.3f}", ha="center",
                va="bottom", fontsize=9, rotation=90)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(xs, [n["in_degree"] for n in nodes], linestyle="none", marker="D",
             markersize=7, markerfacecolor="black", markeredgecolor="white",
             label="In-degree")
    ymax = max(n["in_degree"] for n in nodes)
    ax2.set_ylim(-0.15, ymax * 1.6 + 1)
    ax2.set_yticks(range(0, ymax + 1))
    ax2.set_ylabel("In-degree (count of companies)")
    bar_proxy = matplotlib.patches.Patch(facecolor=C_LIGHT, hatch="//",
                                         edgecolor="black", label="PageRank")
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend([bar_proxy] + h2, ["PageRank (bars)", "In-degree (diamonds)"],
              loc="upper right", frameon=True)
    ax.set_title(
        f"Mention graph PageRank ({len(g['nodes'])} companies, "
        f"{g['params']['n_edges']} edges)\n"
        f"Spearman rho: PageRank vs in-degree = {cor['pagerank_vs_indegree_spearman']:.2f}; "
        f"authority vs in-degree = {cor['authority_vs_indegree_spearman']:.2f}",
        fontsize=9.5)
    save(fig, "fig_pagerank.png")


# --------------------------------------------------------------------- fig 7
def fig_abstention():
    so = load("eval/out/abstention_scoreonly.json")
    mv = load("eval/out/abstention_eval.json")
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    fig, ax = plt.subplots(figsize=(W, 4.0))
    width = 0.36
    xs = list(range(3))
    v1 = [so["test"][k] for k in metrics]
    v2 = [mv["test"][k] for k in metrics]
    b1 = ax.bar([x - width / 2 for x in xs], v1, width, label="Score-only rule")
    style_bar(b1, C_LIGHT, "//")
    b2 = ax.bar([x + width / 2 for x in xs], v2, width, label="Model verdict")
    style_bar(b2, C_DARK, "")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (fraction)")
    ax.set_xlabel("Abstention metric (positive class = should abstain)")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n = so["n"]["test"]
    ax.set_title(f"Abstention rules on test items (n = {n})")
    ax.legend(loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.0))
    t1, t2 = so["test"], mv["test"]
    ax.text(0.0, -0.17,
            f"Score-only: tp={t1['tp']} fp={t1['fp']} fn={t1['fn']} tn={t1['tn']}"
            f"      Model verdict: tp={t2['tp']} fp={t2['fp']} fn={t2['fn']} tn={t2['tn']}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.grid(axis="y", color="#cccccc", linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig_abstention.png")


# --------------------------------------------------------------------- fig 8
def fig_speed_quality():
    c = load("eval/out/cost_table.json")
    r = load("eval/out/retrieval_test.json")
    systems = [s for s in ["tfidf", "tfidf-champ-200", "tfidf-champ-50"]
               if s in c and s in r["overall"]]
    if len(systems) < 2:
        print("skipped fig_speed_quality: champion-list data not found")
        return
    marks = {"tfidf": ("o", "white"), "tfidf-champ-200": ("s", "#9dc3e6"),
             "tfidf-champ-50": ("^", "black")}
    fig, ax = plt.subplots(figsize=(W, 3.8))
    for s in systems:
        lat = c[s]["p50_ms"]
        up = c[s]["p95_ms"] - lat
        nd = r["overall"][s]["nDCG@10"]
        ap = r["overall"][s]["AP"]
        mk, fc = marks[s]
        ax.errorbar(lat, nd, xerr=[[0], [up]], fmt=mk, color="black",
                    markerfacecolor=fc, markersize=10, capsize=3,
                    elinewidth=1.0,
                    label=f"{LABELS[s]}: nDCG@10 {nd:.3f}, AP {ap:.3f}")
    ax.set_xlim(0, max(c[s]["p95_ms"] for s in systems) * 1.15)
    ax.set_xlabel("Query latency, p50 (ms); bar extends to p95")
    ax.set_ylabel("nDCG@10 (test set)")
    ax.set_ylim(0.40, 0.70)
    ax.set_title(f"TF-IDF champion list: latency vs quality "
                 f"({c['n_passages']:,} passages)")
    ax.grid(color="#cccccc", linewidth=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=True)
    save(fig, "fig_speed_quality.png")


def main():
    for fn in (fig_ndcg_systems, fig_pr_curves, fig_ndcg_by_type,
               fig_rag_ratings, fig_entity_accuracy, fig_pagerank,
               fig_abstention, fig_speed_quality):
        fn()
    print("Wrote:")
    for p in OUTPUTS:
        print("  " + os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
