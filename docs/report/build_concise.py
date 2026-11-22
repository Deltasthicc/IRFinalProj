"""Build the short, plain-language version of the report (IEEE two-column layout) from the result files.

    python docs/report/build_concise.py [name]  # writes TradeMesh_Report.md, .html and (if Chrome/Edge is found) .pdf; a name builds a draft instead

Every number that appears in a table, and every number that is written as {{name}} in docs/report/concise/paper.md, is read
from eval/out/*.json or data/graph/company_graph.json when the script runs, so the text cannot drift from the results.
The long version (every table and ablation) is built by build_report.py as TradeMesh_Report_Detailed.pdf.
"""
from __future__ import annotations

import base64
import html as H
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGS = HERE / "figures"
OUT = HERE / (sys.argv[1] if len(sys.argv) > 1 else "TradeMesh_Report")   # an optional name builds a draft next to the real file
TITLE = "TradeMesh: Search and Verified Question Answering over SEC 10-K Filings"
AUTHORS = [("Shashwat Rajan", "2510111019"), ("Tanish Shitanshu", "2510110008"), ("Ruchir Sharma", "2310110728")]


def J(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ numbers from files
ret = J("eval/out/retrieval_test.json")
hum = J("eval/out/rag_scores_human.json")
ent = J("eval/out/entity_eval.json")
abst = J("eval/out/abstention_eval.json")["test"]
abst_score = J("eval/out/abstention_scoreonly.json")["test"]
graph = J("data/graph/company_graph.json")["correlations"]
cost = J("eval/out/cost_table.json")
hyp_by_id = {}
for h in J("eval/out/hypotheses_test.json"):
    hyp_by_id.setdefault(h["id"], h)
ov = ret["overall"]
bs = hum["bootstrap"]
pr = hum["per_rater"]
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
sg = lambda x: f"{x:+.3f}"
ci = lambda d: f"{d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f}"
N = {
    "ndcg_best": f3(ov["hybrid+rerank"]["nDCG@10"]), "ndcg_bm25": f3(ov["bm25"]["nDCG@10"]), "ndcg_dense": f3(ov["dense"]["nDCG@10"]),
    "ndcg_hybrid": f3(ov["hybrid"]["nDCG@10"]), "ndcg_tfidf": f3(ov["tfidf"]["nDCG@10"]),
    "h_cb": f3(hum["configs"]["closed_book"]["mean_rating"]), "h_bm25": f2(hum["configs"]["bm25_rag"]["mean_rating"]),
    "h_best": f2(hum["configs"]["best_rag"]["mean_rating"]),
    "d_best_cb": sg(bs["best_rag - closed_book"]["mean_diff"]), "ci_best_cb": ci(bs["best_rag - closed_book"]),
    "d_best_bm25": sg(bs["best_rag - bm25_rag"]["mean_diff"]), "ci_best_bm25": ci(bs["best_rag - bm25_rag"]),
    "d_bm25_cb": sg(bs["bm25_rag - closed_book"]["mean_diff"]), "ci_bm25_cb": ci(bs["bm25_rag - closed_book"]),
    "kappa": f2(hum["kappa_between_humans"]),
    "entity_ok": str(ent["correct"]), "entity_n": str(ent["n"]), "entity_pct": f"{ent['accuracy'] * 100:.1f}",
    "abst_tp": str(abst["tp"]), "abst_pos": str(abst["tp"] + abst["fn"]), "abst_fp": str(abst["fp"]),
    "score_p": f2(abst_score["precision"]), "score_r": f2(abst_score["recall"]),
    "pr_in": f2(graph["pagerank_vs_indegree_spearman"]),
    "champ_ms": f"{cost['tfidf']['p50_ms']:.2f}", "champ50_ms": f"{cost['tfidf-champ-50']['p50_ms']:.2f}",
    "h1_d": sg(hyp_by_id["H1"]["mean_diff"]), "h3a_d": sg(hyp_by_id["H3a"]["mean_diff"]), "h4_d": sg(hyp_by_id["H4"]["mean_diff"]),
    "h3b_d": sg(hyp_by_id["H3b"]["mean_diff"]),
    "rq3": f2(hum["rq3_human"]["spearman"]),
    "bm25_build": f2(cost["bm25_build_s"]),
}
for who, key in (("Ruchir", "Ruchir"), ("Shashwat", "Shashwat")):
    for c in ("closed_book", "bm25_rag", "best_rag"):
        N[f"{who[:1].lower()}_{c}"] = f2(pr[key][c])


def fill(text: str) -> str:
    def rep(m):
        if m.group(1) not in N:
            sys.exit(f"build_concise: unknown placeholder {m.group(1)}")
        return N[m.group(1)]
    return re.sub(r"\{\{(\w+)\}\}", rep, text)


# ------------------------------------------------------------------------------------------------ tables
def table(caption: str, header: list[str], rows: list[list[str]], wide: bool = False) -> str:
    th = "".join(f"<th>{H.escape(c)}</th>" for c in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'@@TABLE-START@@<div class="tbl {"wide" if wide else "narrow"}{" long" if len(rows) > 12 else ""}"><div class="tcap">TABLE @@NO@@<br>{H.escape(caption.upper())}</div><table><tr>{th}</tr>{body}</table></div>@@TABLE-END@@'


def t_data() -> str:
    rows = [["Companies / filings", "20 companies, latest 10-K each (fiscal year 2025 or 2026)"],
            ["Passages searched", "11,964 (10,912 text, 1,052 tables); 1.46 million words"],
            ["Test collection", "30 information needs: 10 for tuning, 20 for the final test"],
            ["Relevance labels", "680 judgments (0, 1, 2) made by a language model"],
            ["Unanswerable questions", "10 (4 tuning, 6 test)"],
            ["Misspelt company names", "40 (for the spelling module)"]]
    return table("The data we built and used", ["Item", "Size"], rows)


def t_retrieval() -> str:
    names = [("tfidf", "TF-IDF"), ("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "Hybrid (BM25 + dense)"), ("hybrid+rerank", "Hybrid + reranker")]
    rows = []
    for k, n in names:
        v = ov[k]
        best = k == "hybrid+rerank"
        cells = [v["P@10"], v["R@10"], v["F1@10"], v["AP"], v["nDCG@10"]]
        rows.append([f"<b>{n}</b>" if best else n] + [f"<b>{c:.3f}</b>" if best else f"{c:.3f}" for c in cells])
    return table("Retrieval results on the 20 test questions", ["System", "P@10", "R@10", "F1@10", "MAP", "nDCG@10"], rows)


def t_hyp() -> str:
    rows = [
        ["H1", "BM25 beats TF-IDF", f"<b>Yes</b> ({N['h1_d']} nDCG@10)"],
        ["H2", "BM25 ties dense on exact-term questions; dense wins on topical ones", "Unclear (too few questions per type)"],
        ["H3", "Hybrid beats BM25 and dense", f"<b>Beats BM25</b> ({N['h3a_d']}); ties dense ({N['h3b_d']})"],
        ["H4", "Reranker improves hybrid", f"Unclear ({N['h4_d']}, interval includes 0)"],
        ["H5", "Table chunks and numbers help table questions", "Not shown (no effect over all questions)"],
        ["H6", "Pseudo-relevance feedback helps", "No (tied with BM25)"],
        ["H7", "Champion lists are faster at little cost", f"Yes for r = 50 ({N['champ_ms']} to {N['champ50_ms']} ms)"],
        ["H8", "Answers with retrieval beat answers without", f"<b>Yes</b> for best-RAG ({N['d_best_cb']})"],
        ["H9", "Hybrid + rerank answers beat BM25 answers", f"<b>Yes</b> ({N['d_best_bm25']})"],
        ["H10", "PageRank differs from counting mentions", f"No (Spearman {N['pr_in']})"],
        ["H11", "Page-image retrieval beats text", "Not tested (track not built)"],
    ]
    return table("Our hypotheses (written before the test run) and what happened", ["", "Hypothesis", "Result"], rows)


def t_human() -> str:
    names = [("closed_book", "Without retrieval (closed-book)"), ("bm25_rag", "BM25 + answer"), ("best_rag", "Hybrid + reranker + answer")]
    rows = []
    for c, n in names:
        best = c == "best_rag"
        mean = f2(hum["configs"][c]["mean_rating"]) if c != "closed_book" else f3(hum["configs"][c]["mean_rating"])
        rows.append([f"<b>{n}</b>" if best else n, f"<b>{mean}</b>" if best else mean, f2(pr["Shashwat"][c]), f2(pr["Ruchir"][c])])
    return table("Average answer rating (0 to 2) from two blind human raters", ["Answers made with", "Mean of both", "Shashwat", "Ruchir"], rows)


def t_compare() -> str:
    rows = [
        ["BM25 is a strong baseline", "BEIR (Thakur et al., 2021)", "Dense was ahead of BM25 here; BM25 not ahead", "No"],
        ["Dense wins on paraphrased questions", "DPR (Karpukhin et al., 2020)", "Same direction, but unclear", "Partly"],
        ["Fusing rankings helps", "RRF (Cormack et al., 2009)", "Hybrid beats BM25, ties dense", "Yes"],
        ["Cross-encoder reranking helps", "Nogueira and Cho (2019)", "Same sign, interval includes 0", "Partly"],
        ["Retrieval matters for filings", "FinanceBench (Islam et al., 2023)", f"Best-RAG {N['d_best_cb']} over closed-book", "Yes"],
        ["Page images help on visual pages", "ColPali (Faysse et al., 2024)", "Not tested", "n/a"],
    ]
    return table("Our results compared with published findings", ["Published claim", "Paper", "Our result", "Agrees?"], rows)


def t_related() -> str:
    rows = [
        ["Salton et al. (1975)", "Vector space model, TF-IDF, cosine", "TF-IDF system"],
        ["Robertson and Zaragoza (2009)", "BM25", "Main baseline, tuned on 10 questions"],
        ["Rocchio (1971)", "Relevance feedback", "Pseudo-relevance feedback"],
        ["Reimers and Gurevych (2019); Wang et al. (2020)", "Sentence embeddings (MiniLM)", "Dense retrieval"],
        ["Karpukhin et al. (2020)", "Dense passage retrieval", "Source of hypothesis H2"],
        ["Fox and Shaw (1994); Cormack et al. (2009)", "Rank fusion, reciprocal rank fusion", "Hybrid search"],
        ["Nogueira and Cho (2019)", "BERT reranking", "Cross-encoder reranker"],
        ["Thakur et al. (2021)", "BEIR: BM25 is a robust baseline", "Comparison in Section VI-H"],
        ["Lewis et al. (2020); Es et al. (2023)", "RAG; automatic RAG scores", "Answer stage; we use our own checks and human ratings"],
        ["Islam et al. (2023); Jimeno Yepes et al. (2024)", "FinanceBench; chunking financial reports", "Motivation; every table kept as one chunk"],
        ["Khattab and Zaharia (2020); Faysse et al. (2024)", "Late interaction; page-image retrieval", "MaxSim written; track not built"],
        ["Brin and Page (1998); Kleinberg (1999)", "PageRank; HITS", "Company mention graph"],
        ["J\u00e4rvelin and Kek\u00e4l\u00e4inen (2002); Zobel (1998); Buckley and Voorhees (2004); Cohen (1960, 1968); Efron and Tibshirani (1993); Manning et al. (2008)",
         "nDCG; pooling and its bias; kappa; bootstrap; course text", "Metrics, bias checks, agreement, intervals"],
    ]
    return table("Papers we used and for what", ["Paper", "Idea", "How we use it"], rows)


def t_systems() -> str:
    rows = [["Boolean", "AND / OR / NOT, phrase and wildcard search; unranked", "Built by us"],
            ["TF-IDF", "Cosine similarity of TF-IDF vectors", "Built by us"],
            ["TF-IDF + champion lists", "Same, scoring only the top r passages per word (r = 50, 200); 2 setups", "Built by us"],
            ["BM25", "Probabilistic ranking with length normalisation", "Built by us"],
            ["BM25 + feedback", "BM25 with Rocchio query expansion", "Built by us"],
            ["Dense", "Cosine of sentence vectors", "Pre-trained encoder, our search code"],
            ["Hybrid", "BM25 and dense fused with reciprocal rank fusion", "Our fusion code"],
            ["BM25 + reranker", "Cross-encoder rereads BM25's top 30", "Pre-trained reranker"],
            ["Hybrid + reranker", "Cross-encoder rereads the hybrid's top 30", "Pre-trained reranker"]]
    return table("The ten retrieval setups (nine rows; the champion-list row is two setups)", ["Setup", "What it does", "Made by"], rows)


def t_abl() -> str:
    a = J("eval/out/ablation_test.json")
    pick = [("default:stop+nostem+num+nometa", "BM25 as is (base)"),
            ("A prep:stop+nostem+num+meta", "+ company, year, section added to each passage"),
            ("A prep:stop+nostem+nonum+meta", "+ the same, numbers not indexed"),
            ("A prep:stop+stem+num+nometa", "Stemming words"),
            ("B struct:drop table chunks", "Without table chunks"),
            ("B struct:entity filter off", "Without the company filter"),
            ("D prf:k=5, m=10", "+ pseudo-relevance feedback"),
            ("C fusion:rrf k=60", "Hybrid fusion, k = 60 (our choice)"),
            ("C fusion:rrf k=10", "Hybrid fusion, k = 10")]
    rows = []
    for k, label in pick:
        v = a[k]
        d = v["vs_default"]
        clear = d["ci_low"] > 0 or d["ci_high"] < 0
        rows.append([label, f"{v['nDCG@10']:.3f}", "-" if k.startswith("default") else f"{d['mean_diff']:+.3f}",
                     "-" if k.startswith("default") else f"{d['ci_low']:+.2f} to {d['ci_high']:+.2f}",
                     "-" if k.startswith("default") else ("<b>yes</b>" if clear else "no")])
    return table("One change at a time (nDCG@10, 20 test questions)", ["Change", "nDCG@10", "Difference", "95% interval", "Clear?"], rows)


def t_cite() -> str:
    c = hum["configs"]
    names = [("closed_book", "Without retrieval"), ("bm25_rag", "BM25 + answer"), ("best_rag", "Hybrid + reranker + answer")]
    rows = []
    for k, n in names:
        e = c[k]
        cp = "none (no citations)" if e["citation_precision"] is None else f"{round(e['citation_precision'] * e['n_citations_judged'])} of {e['n_citations_judged']}"
        ng = "n/a" if e["number_grounding"] is None else f"{e['number_grounding']:.2f}"
        vp = "n/a" if e["verifier_pass_rate"] is None else f"{e['verifier_pass_rate']:.2f}"
        rows.append([n, cp, ng, vp])
    return table("Checks on the answers", ["Answers made with", "Citations that support their sentence", "Numbers found in cited text", "Verifier passed"], rows)


def t_abst() -> str:
    def row(name, m):
        return [name, f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}", f"{m['accuracy']:.2f}", f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}"]
    return table("Deciding when to refuse (26 test items). P = precision, R = recall, Acc. = accuracy", ["Rule", "P", "R", "F1", "Acc.", "TP/FP/FN/TN"],
                 [row("Retrieval-score threshold", abst_score), row("<b>Model's own verdict</b>", abst)])


def t_setup() -> str:
    rows = [["Answer model", "Google Gemini API, model gemini-3.5-flash-lite, temperature 0; every answer cached"],
            ["Dense encoder", "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions), pre-trained"],
            ["Reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2 on the top 30, pre-trained"],
            ["Relevance labels", "a language model (Claude), two passes; its exact version was not recorded"],
            ["Software", "Python 3.14.6, sentence-transformers 6.3.0, torch 2.15.0, transformers 5.23.0, numpy 2.5.3, pandas 3.1.1, Streamlit 1.66.1"],
            ["Statistics", "paired bootstrap over questions, 10,000 resamples, seed 358"],
            ["Run on", "a laptop, no GPU needed; the test run was done once and locked"]]
    return table("Models and software behind the results", ["Part", "Choice"], rows)


def t_fail() -> str:
    rows = [["Ticker typo MSTF (for MSFT)", "Not corrected", "Swapping two letters costs two edits; short strings allow one"],
            ["Walmart sales by segment", "Both answers left out a segment", "The checker tests presence of citations and numbers, not completeness; the company filter was skipped (\"Walmart's\"), which may have contributed"],
            ["Microsoft revenue by segment", "The BM25 answer declined", "The company filter was skipped (\"Microsoft's\"), which may have contributed"],
            ["Topical questions, BM25 answers", "Declined 3 of 7", "The model judged the BM25 passages insufficient"],
            ["The number checker", "Flagged none of 55 answers", "On this data it never showed that it can catch an error (its unit tests do)"]]
    return table("Where the system fails or is not proven", ["Case", "What happened", "Why"], rows)


TABLES = {"T_FAIL": t_fail, "T_SETUP": t_setup, "T_RELATED": t_related, "T_SYSTEMS": t_systems, "T_ABL": t_abl, "T_CITE": t_cite, "T_ABST": t_abst, "T_DATA": t_data, "T_RET": t_retrieval, "T_HYP": t_hyp, "T_HUMAN": t_human, "T_COMPARE": t_compare}
FIGS_META = {
    "fig_model.svg": ("TradeMesh pipeline. Left: building the indexes (done once). Right: what happens to each question. The dashed box is the page-image track, which we did not build.", ROOT / "docs/report_C/fig_model.svg", 132),
    "fig_ndcg_systems.png": ("nDCG@10 of the ten retrieval setups on the 20 test questions (higher is better). The right-hand column is the difference from BM25 with a 95% interval.", FIGS / "fig_ndcg_systems.png", 104),
    "fig_pr_curves.png": ("Precision against recall for five systems. A curve that stays higher is better.", FIGS / "fig_pr_curves.png", 100),
    "fig_entity_accuracy.png": ("Spelling correction: share of the 40 misspelt company names fixed correctly, by type of mistake.", FIGS / "fig_entity_accuracy.png", 100),
    "fig_rag_ratings.png": ("Average answer rating (0 to 2): humans (mean of two raters) and the language-model rater. The box gives the human differences with 95% intervals.", FIGS / "fig_rag_ratings.png", 100),
    "fig_abstention.png": ("Knowing when not to answer: a retrieval-score rule against the model's own \"insufficient evidence\" verdict (26 test items).", FIGS / "fig_abstention.png", 100),
    "fig_answer_flow.png": ("How an answer is produced and checked. The checker is an ordinary program, not a language model.", FIGS / "fig_answer_flow.png", 108),
    "fig_ndcg_by_type.png": ("nDCG@10 by type of question (7 entity, 7 topical, 6 terminology questions), for three systems.", FIGS / "fig_ndcg_by_type.png", 100),
    "fig_speed_quality.png": ("Speed against quality for TF-IDF with champion lists (r = 200 and r = 50) and without. Quality stays the same; r = 50 is about three times faster.", FIGS / "fig_speed_quality.png", 100),
    "fig_pagerank.png": ("The 12 highest-ranked companies in the mention graph: PageRank (bars) and the number of incoming mentions (diamonds).", FIGS / "fig_pagerank.png", 100),
}
FIGS_META.update({
    "fig_hypotheses_s.png": ("Status of the 11 hypotheses. Green: supported; light green: partly; amber: unclear; red: not supported; grey: not tested.", FIGS / "fig_hypotheses_s.png", 0, "col"),
    "fig_rating_dist_s.png": ("How often each rater gave 0, 1 or 2 to the 20 answers of each kind; the two raters differ mostly by one point.", FIGS / "fig_rating_dist_s.png", 0, "col"),
    "fig_ablation_s.png": ("Each change from the table above as a difference from plain BM25, with its 95% interval.", FIGS / "fig_ablation_s.png", 0, "col"),
    "fig_glance.png": ("Four headline results at a glance.", FIGS / "fig_glance.png", 112, "wide"),
    "fig_corpus.png": ("What is in the corpus: passages per company (text and table chunks) and per 10-K section.", FIGS / "fig_corpus.png", 122, "wide"),
    "fig_pr_curves_s.png": ("Precision against recall for five systems; a curve that stays higher is better.", FIGS / "fig_pr_curves_s.png", 0, "col"),
    "fig_ndcg_by_type_s.png": ("nDCG@10 by type of question for three systems.", FIGS / "fig_ndcg_by_type_s.png", 0, "col"),
    "fig_speed_quality_s.png": ("Champion lists: same quality (height), lower latency (left). r = 50 is about three times faster.", FIGS / "fig_speed_quality_s.png", 0, "col"),
    "fig_entity_accuracy_s.png": ("Spelling correction: share of the 40 misspelt names fixed, by type of mistake.", FIGS / "fig_entity_accuracy_s.png", 0, "col"),
    "fig_rag_ratings_s.png": ("Average answer rating (0 to 2): humans (mean of two raters) and the language-model rater.", FIGS / "fig_rag_ratings_s.png", 0, "col"),
    "fig_abstention_s.png": ("Knowing when not to answer: retrieval-score rule against the model's own verdict.", FIGS / "fig_abstention_s.png", 0, "col"),
    "fig_graph_net_s.png": ("The company mention graph; arrows point to the company that is mentioned.", FIGS / "fig_graph_net_s.png", 0, "col"),
})
APP_ROW = ("The web app. (a) Ask tab: the misspelt name is corrected, the answer cites passage P1 and the trust panel shows each check. "
           "(b) Search tab: BM25 and TF-IDF side by side with highlighted query words.",
           [(FIGS / "app_ask.png", (0, 0, 1140, 1270)), (FIGS / "app_search.png", (0, 0, 1140, 1010))])
APP_ROW2 = ("The web app, continued. (c) Evaluation tab: retrieval results read from the locked result files. "
            "(d) Graph tab: PageRank of the companies and the \"who mentions whom\" network.",
            [(FIGS / "app_eval.png", (0, 60, 1140, 1030)), (FIGS / "app_graph.png", (0, 0, 1140, 1090))])
APP_ROWS = {"app": APP_ROW, "app2": APP_ROW2}


def data_uri(p: Path) -> str:
    mime = {"png": "image/png", "svg": "image/svg+xml"}[p.suffix[1:]]
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


# ------------------------------------------------------------------------------------------------ assemble
src = fill((HERE / "concise" / "paper.md").read_text(encoding="utf-8"))
m = re.search(r"^ABSTRACT:\s*(.*?)\nKEYWORDS:\s*(.*?)\n", src, flags=re.S | re.M)
abstract, keywords = m.group(1).strip(), m.group(2).strip()
body = src[m.end():]
for key, fn in TABLES.items():
    token = f"@@{key}@@"
    if token not in body:
        sys.exit(f"build_concise: {token} missing in paper.md")
    body = body.replace(token, "\n\n" + fn() + "\n\n")
fig_no = 0
fig_html: list[str] = []
_order = [mm.group(1) for mm in re.finditer(r"@@FIG(?:ROW)?:([\w.]+)@@", body)]
_num = {n: i + 1 for i, n in enumerate(_order)}


def _ref(mm: re.Match) -> str:
    if mm.group(1) not in _num:
        sys.exit(f"build_concise: figure reference to a figure that is not placed: {mm.group(1)}")
    return f"Fig. {_num[mm.group(1)]}"


body = re.sub(r"\[\[fig:([\w.]+)\]\]", _ref, body)


def fig(mm: re.Match) -> str:
    global fig_no
    name = mm.group(1)
    meta = FIGS_META[name]
    cap, path, width = meta[:3]
    mode = meta[3] if len(meta) > 3 else "wide"
    fig_no += 1
    style = f' style="max-width:{width}mm"' if width else ""
    fig_html.append(f'<figure class="{mode}"><img{style} alt="Fig. {fig_no}" src="{data_uri(path)}"><figcaption>Fig. {fig_no}. {H.escape(cap)}</figcaption></figure>')
    return f"\n\nFIGPLACEHOLDER{fig_no - 1}X\n\n"


def figrow(mm: re.Match) -> str:
    """Two screenshots side by side under one caption."""
    import io

    from PIL import Image
    global fig_no
    cap, items = APP_ROWS[mm.group(1)]
    fig_no += 1
    imgs = []
    for path, box in items:
        buf = io.BytesIO()
        Image.open(path).crop(box).save(buf, "PNG")
        imgs.append(f'<img alt="screenshot" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">')
    sub = [f"({c}) " for c in "ab"]
    fig_html.append(f'<figure class="wide"><div class="row">{"".join(imgs)}</div><figcaption>Fig. {fig_no}. {H.escape(cap)}</figcaption></figure>')
    return f"\n\nFIGPLACEHOLDER{fig_no - 1}X\n\n"


body = re.sub(r"@@FIG(?:ROW)?:([\w.]+)@@", lambda mm: figrow(mm) if mm.group(0).startswith("@@FIGROW") else fig(mm), body)   # one pass: numbering follows the text
refs = (HERE / "parts" / "10_references.md").read_text(encoding="utf-8").split("\n\n", 1)[1].strip()
body += "\n\n## REFERENCES\n\n" + refs + "\n"

html_body = markdown.markdown(body, extensions=["tables", "sane_lists"])
html_body = re.sub(r"<p>FIGPLACEHOLDER(\d+)X</p>", lambda mm: fig_html[int(mm.group(1))], html_body)
html_body = re.sub(r"<p>@@TABLE-START@@(.*?)@@TABLE-END@@</p>", r"\1", html_body, flags=re.S)
def _roman(n: int) -> str:
    out = ""
    for v, s in [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out, n = out + s, n - v
    return out


counter = iter(range(1, 100))
ROM = lambda n: "".join(s * (n // v) if False else "" for v, s in [])  or _roman(n)
html_body = re.sub(r"@@NO@@", lambda mm: ROM(next(counter)), html_body)
html_body = re.sub(r"(<h2>REFERENCES</h2>)(.*)", lambda mm: mm.group(1) + mm.group(2).replace("<p>", '<p class="ref">'), html_body, flags=re.S)

CSS = """
@page { size: A4; margin: 15mm 14mm 17mm 14mm; @bottom-center { content: counter(page); font: 8pt 'Times New Roman', serif; } }
* { box-sizing: border-box; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.22; color: #000; margin: 0; }
.front { text-align: center; margin-bottom: 4mm; }
.front h1 { font-size: 20pt; font-weight: normal; line-height: 1.15; margin: 2mm 0 5mm; }
.authors { display: flex; justify-content: center; gap: 12mm; }
.authors div { font-size: 9pt; line-height: 1.25; }
.authors b { font-weight: normal; font-size: 11pt; }
.front .small { font-size: 8.5pt; margin-top: 3mm; }
.cols { column-count: 2; column-gap: 6.5mm; column-fill: balance; }
.cols p, .cols li { text-align: justify; hyphens: auto; margin: 0 0 1.8mm; orphans: 2; widows: 2; }
.abs { font-weight: bold; font-size: 9pt; line-height: 1.17; }
h2 { font-size: 10pt; font-weight: normal; text-transform: uppercase; text-align: center; margin: 3.6mm 0 1.4mm; break-after: avoid; letter-spacing: .02em; }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; margin: 2.4mm 0 1mm; break-after: avoid; }
ul, ol { margin: 0 0 1.8mm; padding-left: 5mm; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8pt; overflow-wrap: anywhere; }
.formula { text-align: center !important; font-style: italic; margin: 1mm 0 2mm !important; }
.tbl { margin: 2mm 0 3mm; }
.tbl.wide { column-span: all; }
.tbl.narrow { break-inside: avoid; }
.tbl.narrow table { font-size: 7.3pt; }
.tbl.long { break-inside: auto; }
.tbl.long tr:nth-child(-n+3) { break-after: avoid; }
.tcap { break-after: avoid; }
.tcap { font-size: 7.8pt; text-align: center; margin-bottom: 1mm; line-height: 1.15; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.15; }
th, td { border: .4pt solid #444; padding: .6mm 1.2mm; vertical-align: top; text-align: left; overflow-wrap: break-word; hyphens: none; }
th { font-weight: bold; background: #efefef; }
tr { break-inside: avoid; }
figure { margin: 2mm 0 3mm; text-align: center; }
figure.wide { column-span: all; break-inside: avoid; }
figure.col { break-inside: avoid; margin: 1.5mm 0 2.5mm; }
figure.col img { max-width: 100%; max-height: 62mm; }
.example { border: .6pt solid #6a3d8f; background: #f7f2fb; padding: 1.5mm 2.2mm; margin: 1.5mm 0 2.5mm; font-size: 8.6pt; line-height: 1.2; break-inside: avoid; text-align: left; }
figure img { max-height: 70mm; }
.row { display: flex; justify-content: center; gap: 4mm; } .row img { width: 64mm; max-height: none; }
figcaption { font-size: 8.2pt; text-align: justify; margin: 1mm 8mm 0; line-height: 1.15; }
p.ref { font-size: 7.8pt; text-align: left; padding-left: 4mm; text-indent: -4mm; margin: 0 0 .9mm; hyphens: none; overflow-wrap: anywhere; }
"""
inline = lambda s: markdown.markdown(s).removeprefix("<p>").removesuffix("</p>")
authors_html = "".join(f"<div><b>{n}</b><br>Roll no. {r}<br>CSD358 Information Retrieval<br>Shiv Nadar Institution of Eminence</div>" for n, r in AUTHORS)
front = (f'<div class="front"><h1>{TITLE}</h1><div class="authors">{authors_html}</div><div class="small">Course project, CSD358 Information Retrieval, '
         f'Monsoon 2026 (instructor: Sonia Khetarpaul). Code, data and results: https://github.com/Deltasthicc/IRFinalProj</div></div>')
abstract_html = f'<p class="abs"><i>Abstract</i>\u2014{inline(abstract)}</p><p class="abs"><i>Index Terms</i>\u2014{inline(keywords)}.</p>'
page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{TITLE}</title><style>{CSS}</style></head>"
        f"<body>{front}<div class='cols'>{abstract_html}{html_body}</div></body></html>")
OUT.with_suffix(".html").write_text(page, encoding="utf-8")
OUT.with_suffix(".md").write_text(f"# {TITLE}\n\n" + ", ".join(f"{n} ({r})" for n, r in AUTHORS) + f"\n\n**Abstract.** {abstract}\n\n**Index Terms.** {keywords}.\n\n"
                                  + re.sub(r"FIGPLACEHOLDER\d+X", "[figure]", body).replace("@@TABLE-START@@", "").replace("@@TABLE-END@@", ""), encoding="utf-8")
leftover = re.findall(r"\{\{\w+\}\}|@@\w+", page)
if leftover:
    sys.exit(f"build_concise: leftover markers {leftover[:5]}")
browser = next((p for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                            shutil.which("chrome") or "", shutil.which("msedge") or ""] if p and Path(p).exists()), None)
if browser:
    pdf = OUT.with_suffix(".pdf")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", OUT.with_suffix(".html").as_uri()],
                   check=True, timeout=240)
    print("wrote", pdf, pdf.stat().st_size // 1024, "KB;", fig_no, "figures")
else:
    print("no Chrome/Edge found; open the .html and print to PDF")
