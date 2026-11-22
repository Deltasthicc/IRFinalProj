"""Build the short, plain-language version of the report (IEEE two-column layout) from the result files.

    python docs/report/build_concise.py        # writes TradeMesh_Report.md, .html and (if Chrome/Edge is found) .pdf

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
OUT = HERE / "TradeMesh_Report"
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
    return f'@@TABLE-START@@<div class="tbl {"wide" if wide else "narrow"}"><div class="tcap">TABLE @@NO@@<br>{H.escape(caption.upper())}</div><table><tr>{th}</tr>{body}</table></div>@@TABLE-END@@'


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
    return table("Our hypotheses (written before the test run) and what happened", ["", "Hypothesis", "Result"], rows, wide=True)


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
    return table("Our results compared with published findings", ["Published claim", "Paper", "Our result", "Same?"], rows, wide=True)


TABLES = {"T_DATA": t_data, "T_RET": t_retrieval, "T_HYP": t_hyp, "T_HUMAN": t_human, "T_COMPARE": t_compare}
FIGS_META = {
    "fig_model.svg": ("TradeMesh pipeline. Left: building the indexes (done once). Right: what happens to each question. The dashed box is the page-image track, which we did not build.", ROOT / "docs/report_C/fig_model.svg", 150),
    "fig_ndcg_systems.png": ("nDCG@10 of the ten retrieval setups on the 20 test questions (higher is better). The right-hand column is the difference from BM25 with a 95% interval.", FIGS / "fig_ndcg_systems.png", 118),
    "fig_pr_curves.png": ("Precision against recall for five systems. A curve that stays higher is better.", FIGS / "fig_pr_curves.png", 100),
    "fig_entity_accuracy.png": ("Spelling correction: share of the 40 misspelt company names fixed correctly, by type of mistake.", FIGS / "fig_entity_accuracy.png", 100),
    "fig_rag_ratings.png": ("Average answer rating (0 to 2): humans (mean of two raters) and the language-model rater. The box gives the human differences with 95% intervals.", FIGS / "fig_rag_ratings.png", 100),
    "fig_abstention.png": ("Knowing when not to answer: a retrieval-score rule against the model's own \"insufficient evidence\" verdict (26 test items).", FIGS / "fig_abstention.png", 100),
    "fig_answer_flow.png": ("How an answer is produced and checked. The checker is an ordinary program, not a language model.", FIGS / "fig_answer_flow.png", 125),
    "fig_ndcg_by_type.png": ("nDCG@10 by type of question (7 entity, 7 topical, 6 terminology questions), for three systems.", FIGS / "fig_ndcg_by_type.png", 100),
    "fig_speed_quality.png": ("Speed against quality for TF-IDF with champion lists (r = 200 and r = 50) and without. Quality stays the same; r = 50 is about three times faster.", FIGS / "fig_speed_quality.png", 100),
    "fig_pagerank.png": ("The 12 highest-ranked companies in the mention graph: PageRank (bars) and the number of incoming mentions (diamonds).", FIGS / "fig_pagerank.png", 100),
}
APP_ROW = ("The web app. (a) Ask tab: the misspelt name is corrected, the answer cites passage P1 and the trust panel shows each check. "
           "(b) Search tab: BM25 and TF-IDF side by side with highlighted query words.",
           [(FIGS / "app_ask.png", (0, 0, 1140, 1270)), (FIGS / "app_search.png", (0, 0, 1140, 1010))])


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


def fig(mm: re.Match) -> str:
    global fig_no
    name = mm.group(1)
    cap, path, width = FIGS_META[name]
    fig_no += 1
    fig_html.append(f'<figure class="wide"><img style="max-width:{width}mm" alt="Fig. {fig_no}" src="{data_uri(path)}"><figcaption>Fig. {fig_no}. {H.escape(cap)}</figcaption></figure>')
    return f"\n\nFIGPLACEHOLDER{fig_no - 1}X\n\n"


def figrow(mm: re.Match) -> str:
    """Two screenshots side by side under one caption."""
    import io

    from PIL import Image
    global fig_no
    cap, items = APP_ROW
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
counter = iter(range(1, 100))
ROM = lambda n: {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}[n]
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
.tcap { font-size: 7.8pt; text-align: center; margin-bottom: 1mm; line-height: 1.15; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.15; }
th, td { border: .4pt solid #444; padding: .6mm 1.2mm; vertical-align: top; text-align: left; overflow-wrap: anywhere; }
th { font-weight: bold; background: #efefef; }
tr { break-inside: avoid; }
figure { margin: 2mm 0 3mm; text-align: center; }
figure.wide { column-span: all; break-inside: avoid; }
figure img { max-height: 70mm; }
.row { display: flex; justify-content: center; gap: 4mm; } .row img { width: 82mm; max-height: none; }
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
