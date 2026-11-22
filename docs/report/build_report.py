"""Assemble the final report from its parts and render it to PDF.

    python docs/report/build_report.py            # writes TradeMesh_Report_IEEE.md, .html and (if Chrome/Edge is found) .pdf

Sources: the new parts in docs/report/parts/, the role drafts in docs/report_B and docs/report_C (renumbered to the
instructor's ten-section layout), the figures in docs/report/figures/ and docs/report_C/fig_model.svg.
Every number in the report is in the source text; this script only renumbers, patches cross-references, inserts
figures and renders. It stops with an error if an expected string is missing, so a changed draft cannot slip through.
"""
from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
PARTS, FIGS = HERE / "parts", HERE / "figures"
B, C = DOCS / "report_B", DOCS / "report_C"
OUT = HERE / "TradeMesh_Report_IEEE"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip("\n")


def patch(text: str, old: str, new: str, what: str) -> str:
    if old not in text:
        sys.exit(f"build_report: expected text not found ({what}): {old!r}")
    return text.replace(old, new)


def demote(text: str, by: int = 1) -> str:
    """Make every heading `by` levels deeper (fenced blocks are not touched; the drafts have none with '#')."""
    return re.sub(r"^(#{1,5}) ", lambda m: "#" * (len(m.group(1)) + by) + " ", text, flags=re.M)


def drop_h1(text: str) -> str:
    return re.sub(r"\A# .*\n+", "", text)


def section(path: Path, heading: str, patches: list[tuple[str, str]] = ()) -> str:
    """A role draft whose H1 becomes the numbered H2 `heading`; its own H2/H3 move one level down."""
    t = read(path)
    for a, b in patches:
        t = patch(t, a, b, path.name)
    return f"## {heading}\n\n" + demote(drop_h1(t))


def clean_roles(t: str) -> str:
    for a, b in [("Role A's `data/companies.json`", "`data/companies.json`"), ("Role C's RAG study", "The RAG study"),
                 ("threshold tau of Role C is", "threshold tau is"), ("(the author of this section)", "(a project member)"), (" (Report B)", ""), (" (Role B)", ""), (" (Role C)", "")]:
        t = t.replace(a, b)
    return t


# ---------------------------------------------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------------------------------------------
parts: list[str] = []
parts.append(read(PARTS / "01_title.md"))
parts.append(read(PARTS / "02_abstract.md"))
parts.append(read(PARTS / "03_introduction.md"))
parts.append(read(PARTS / "04_literature_review.md"))
parts.append(read(PARTS / "05_objective.md"))

# 6 proposed model (figure 1 is the model diagram)
model = read(C / "06_proposed_model.md")
model = patch(model, "![Figure 1: the TradeMesh model](fig_model.svg)\n\n", "@@FIG_MODEL@@\n\n", "model figure")
model = re.sub(r"\n\*Figure 1\. .*\n", "\n", model)          # the caption is regenerated below
parts.append("## 6. Proposed model\n\n" + drop_h1(model))

# 7 methodology
parts.append(read(PARTS / "07_0_methodology_intro.md"))
parts.append(read(PARTS / "07_1_positional_index_boolean_wildcard.md"))
parts.append(read(PARTS / "07_2_tfidf_champion_lists.md"))
rank = read(B / "09_3_to_9_5_ranking_methods.md")
rank = drop_h1(rank)
for n in (3, 4, 5):
    rank = rank.replace(f"## 9.{n} ", f"## 7.{n} ")
rank = patch(rank, "## What is not claimed", "### What is not claimed in sections 7.3 to 7.5", "ranking heading")
parts.append(rank)
parts.append(section(C / "09_6_query_processing.md", "7.6 Query processing, entities, section prior, table chunks",
                     [("section 9.8", "section 7.8")]))
parts.append(read(PARTS / "07_7_company_graph.md").replace("section 9.6", "section 7.6"))
parts.append(section(C / "09_8_rag_verifier_abstention.md", "7.8 RAG, verifier and abstention"))
parts.append(read(PARTS / "07_9_vision.md"))
parts.append(section(B / "09_10_evaluation_protocol.md", "7.10 Evaluation protocol", [("section 10.3", "section 8.4")]))

# 8 experiments
ret = read(B / "10_retrieval_results.md")
ret = drop_h1(ret)
intro8, rest8 = ret.split("## 10.1 ", 1)
rest8 = "## 10.1 " + rest8
rest8 = patch(rest8, "section 10.11", "section 8.12", "ref to RAG study")
rest8 = patch(rest8, "see 10.5)", "see 8.6)", "ref to ablations")
COST_EXTRA = (" It was measured on one laptop, once.\n\n**Storage and start-up.** The processed passages file is 11 MB. The dense embeddings of all 11,964 "
              "passages (384 dimensions, 32-bit) take 18 MB; they are computed on the first start of the application with the ML packages "
              "installed and reused from disk afterwards. The BM25 index is built in memory in 0.64 s. The application starts in about 15 s "
              "with BM25 only and about 50 s with the dense model on a warm laptop; a cold first start took about four minutes. End-to-end "
              "answer latency was not logged, so we give no figure: a live language-model call took from a few seconds to over a minute "
              "(the client waits out the provider's rate limits), and a cached answer returns at once.")
rest8 = patch(rest8, " It was measured on one laptop, once.", COST_EXTRA, "cost paragraph")
for k in range(9, 0, -1):                                    # 10.k -> 8.(k+1); descending so 10.1 cannot hit 10.10
    rest8 = re.sub(rf"^## 10\.{k} ", f"## 8.{k + 1} ", rest8, flags=re.M)
parts.append("## 8. Experimentation and results\n\n" + intro8.strip())
parts.append(read(PARTS / "08_1_datasets.md").replace("{{CORPUS_TABLE}}", read(PARTS / "_corpus_table.md")))
parts.append(rest8)
cmp_ = read(B / "10_paper_comparison.md")
cmp_ = patch(cmp_, "# 10.10 Comparison with published findings", "## 8.11 Comparison with published findings", "cmp heading")
parts.append(clean_roles(cmp_))
parts.append(clean_roles(section(C / "10_rag_results.md", "8.12 RAG study, abstention and entity correction")))

# 9 conclusions
concl = read(C / "11_conclusions_limitations.md")
lim = concl.split("## Limitations", 1)[1].strip("\n")
parts.append(read(PARTS / "09_0_conclusions_head.md"))
parts.append("### 9.2 Limitations\n\n" + lim + "\n" + read(PARTS / "09_2_limitations_extra.md"))
parts.append(read(PARTS / "09_3_future_work.md"))
parts.append(read(PARTS / "10_references.md"))

text = "\n\n".join(p.strip("\n") for p in parts) + "\n"
text = clean_roles(text)
for old, new_ in [("Manning et al. [17]", "Manning et al. (2008)"), ("retrieved text [11]", "retrieved text (Lewis et al. 2020)"),
                  ("even with retrieval [13]", "even with retrieval (Islam et al. 2023)")]:     # numeric citations in the draft
    text = patch(text, old, new_, "numeric citation")

# ---------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------
caps: dict[str, str] = {}
for m in re.finditer(r"\*\*(fig_\w+)\.png\.\*\* (.+)", read(FIGS / "captions.md")):
    caps[m.group(1)] = m.group(2).strip()
MODEL_CAP = ("The TradeMesh model. Offline build (collection, indexes, graph), the two online stages (retrieve, then generate "
             "and check) and the evaluation scripts. The dashed box is the gated vision track, which was not built. Text "
             "retrieval and reranking run on a laptop CPU; generation is one cached language-model call per question.")
placement = [                          # (figure, insert at the end of the section whose heading starts with)
    ("fig_ndcg_systems", "## 8.3 "), ("fig_pr_curves", "## 8.3 "), ("fig_ndcg_by_type", "## 8.4 "),
    ("fig_speed_quality", "## 8.7 "), ("fig_pagerank", "## 7.7 "), ("fig_entity_accuracy", "### RAG study"),
    ("fig_rag_ratings", "### Model-assisted ratings"), ("fig_abstention", "### Section prior"),
]


def data_uri(p: Path) -> str:
    mime = {"png": "image/png", "svg": "image/svg+xml"}[p.suffix[1:]]
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def figure_md(path: Path, cap: str) -> str:
    return f"@@FIG:{path.name}@@{cap}@@END@@"


def insert_before(text: str, heading_prefix: str, block: str, *, at_end_of: bool) -> str:
    """Insert block before the heading that starts with heading_prefix, or at the end of that section."""
    lines = text.split("\n")
    idx = next((i for i, l in enumerate(lines) if l.startswith(heading_prefix)), None)
    if idx is None:
        sys.exit(f"build_report: heading not found for figure: {heading_prefix!r}")
    if not at_end_of:
        lines[idx:idx] = [block, ""]
        return "\n".join(lines)
    level = len(heading_prefix.split(" ")[0])
    j = next((i for i in range(idx + 1, len(lines)) if re.match(rf"^#{{1,{level}}} ", lines[i])), len(lines))
    lines[j:j] = ["", block, ""]
    return "\n".join(lines)


# figures that belong *before* a later heading (they close the previous subsection) vs. at the end of a section
before_heading = {"fig_entity_accuracy", "fig_rag_ratings", "fig_abstention"}
for name, head in placement:
    if name not in caps:
        sys.exit(f"build_report: no caption for {name}")
    block = figure_md(FIGS / f"{name}.png", caps[name])
    # "### RAG study" etc. are demoted to level 3 by section(); find them by their text
    if name in before_heading:
        head = {"fig_entity_accuracy": "### RAG study", "fig_rag_ratings": "#### Model-assisted ratings",
                "fig_abstention": "### Section prior"}[name]
    text = insert_before(text, head, block, at_end_of=name not in before_heading)
text = text.replace("@@FIG_MODEL@@", figure_md(C / "fig_model.svg", MODEL_CAP))

# ---------------------------------------------------------------------------------------------------------------
# IEEE conference layout: Roman-numeral sections (I to VII), lettered subsections, two columns, numbered tables and figures.
# The instructor's section order is kept (introduction, literature review, objective, proposed model, methodology,
# experiments, conclusions, references); only the numbering follows the IEEE convention. Citations stay author-year and the
# reference list keeps the instructor's required format.
# ---------------------------------------------------------------------------------------------------------------
ROMAN = {3: "I", 4: "II", 5: "III", 6: "IV", 7: "V", 8: "VI", 9: "VII"}
AUTHORS = [("Shashwat Rajan", "2510111019"), ("Tanish Shitanshu", "2510110008"), ("Ruchir Sharma", "2310110728")]
TITLE = "TradeMesh: Search and Verified Question Answering over SEC 10-K Filings"
TABLE_CAPTIONS = [
    "Pre-declared hypotheses and their outcomes on the test split",
    "Retrieval systems and the sections that describe them",
    "The filing corpus: 20 companies and 11,964 passages",
    "Relevance judgments and agreement between the two model passes",
    "Retrieval effectiveness on the 20 test needs",
    "Paired bootstrap differences from BM25 with 95% intervals",
    "nDCG@10 by need type",
    "Outcome of the pre-declared retrieval hypotheses",
    "Ablations of preprocessing, structure, fusion and feedback",
    "Latency and quality by retrieval system",
    "Pool-bias check: leave-one-system-out",
    "Observed ordering compared with published findings",
    "Entity correction on 40 misspelt company names",
    "Human ratings of the three answer configurations (mean of two raters)",
    "Human ratings: each rater on their own",
    "Human ratings: paired differences between configurations",
    "Human mean rating by question type (mean of two raters)",
    "Model-assisted ratings of the three answer configurations",
    "Model-assisted ratings: paired differences between configurations",
    "Model-assisted mean rating by question type",
    "Abstention with the retrieval-score rule only",
    "Abstention with the model verdict combined with the score threshold",
]
FIG_RE = re.compile(r"@@FIG:([\w.]+)@@(.*?)@@END@@", re.S)


def roman(n: int) -> str:
    out = ""
    for v, s in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                 (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out, n = out + s, n - v
    return out


def ieeeify(t: str) -> tuple[str, str, str, list[str]]:
    """Return (body markdown, abstract, keywords, log of converted cross-references)."""
    m = re.search(r"^## 2\. Abstract\n\n(.*?)\n\n\*\*Keywords:\*\* (.*?)\n", t, flags=re.M | re.S)
    if not m:
        sys.exit("build_report: abstract or keywords not found")
    abstract, keywords = m.group(1).strip(), m.group(2).strip().rstrip(".")
    t = t[:m.start()] + t[m.end():]
    t = re.sub(r"\A# .*?(?=^## )", "", t, flags=re.M | re.S)                  # the old title block
    valid = {(int(a), int(b)) for a, b in re.findall(r"^#{2,3} (\d+)\.(\d+) ", t, flags=re.M)}
    out = []
    for line in t.split("\n"):
        top = re.match(r"^## (\d+)\. (.*)$", line)
        sub = re.match(r"^#{2,3} (\d+)\.(\d+) (.*)$", line)
        other = re.match(r"^#{3,4} (.*)$", line)
        if top:
            n = int(top.group(1))
            out.append("## REFERENCES" if n == 10 else f"## {ROMAN[n]}. {top.group(2)}")
        elif sub:
            out.append(f"### {chr(64 + int(sub.group(2)))}. {sub.group(3)}")
        elif other:
            out.append(f"#### {other.group(1)}")
        else:
            out.append(line)
    t = "\n".join(out)
    log: list[str] = []

    def ref_a(mm: re.Match) -> str:
        n, k = int(mm.group(1)), int(mm.group(2))
        if (n, k) not in valid:
            return mm.group(0)
        log.append(mm.string[max(0, mm.start() - 40):mm.end() + 25].replace("\n", " "))
        return f"{ROMAN[n]}-{chr(64 + k)}"

    def ref_b(mm: re.Match) -> str:
        n = int(mm.group(2))
        if n not in ROMAN:
            sys.exit(f"build_report: bare section reference to {n} cannot be converted: {mm.group(0)!r}")
        return f"{mm.group(1)} {ROMAN[n]}"

    segs = re.split(r"(`[^`]*`|@@FIG:.*?@@END@@|https?://\S+)", t, flags=re.S)
    for i in range(0, len(segs), 2):
        segs[i] = re.sub(r"(?<![\w.\-+/])(\d{1,2})\.(\d{1,2})(?![\w%]|\.\d)", ref_a, segs[i])
        segs[i] = re.sub(r"\b([Ss]ections?) (\d{1,2})\b(?![\d.]|-)", ref_b, segs[i])
    return "".join(segs), abstract, keywords, log


text, abstract_md, keywords_md, ref_log = ieeeify(text)
md_for_file = (f"# {TITLE}\n\n" + ", ".join(f"{n} ({r})" for n, r in AUTHORS) + "\n\n**Abstract.** " + abstract_md
               + "\n\n**Index Terms.** " + keywords_md + ".\n\n" + FIG_RE.sub(lambda m: f"*[Fig.: {m.group(1)}]* {m.group(2)}", text))
OUT.with_suffix(".md").write_text(md_for_file, encoding="utf-8")
print(f"{len(ref_log)} cross-references converted")
if "--show-refs" in sys.argv:
    for c in ref_log:
        print("   ", c)

count = 0
figs_html: list[str] = []


def inline(md_text: str) -> str:
    return markdown.markdown(md_text).removeprefix("<p>").removesuffix("</p>")


def stash_fig(m: re.Match) -> str:
    """Replace a figure marker by a placeholder paragraph; the figure HTML is put back after markdown has run."""
    global count
    count += 1
    path = (C if m.group(1) == "fig_model.svg" else FIGS) / m.group(1)
    figs_html.append(f'<figure class="wide"><img alt="Fig. {count}" src="{data_uri(path)}">'
                     f"<figcaption>Fig. {count}. {inline(m.group(2))}</figcaption></figure>")
    return "\n\nFIGPLACEHOLDER" + str(count - 1) + "X\n\n"


html_body = markdown.markdown(FIG_RE.sub(stash_fig, text), extensions=["tables", "fenced_code", "sane_lists"])
html_body = re.sub(r"<p>FIGPLACEHOLDER(\d+)X</p>", lambda m: figs_html[int(m.group(1))], html_body)

tables = re.findall(r"<table>.*?</table>", html_body, flags=re.S)
if len(tables) != len(TABLE_CAPTIONS):
    sys.exit(f"build_report: {len(tables)} tables found but {len(TABLE_CAPTIONS)} captions written; update TABLE_CAPTIONS")
table_no = iter(range(len(tables)))


def wrap_table(m: re.Match) -> str:
    k = next(table_no)
    tb = m.group(0)
    cols = tb.split("</tr>")[0].count("<th")
    narrow = cols <= 3 and len(re.sub(r"<[^>]+>", "", tb)) < 1100
    return (f'<div class="tbl {"narrow" if narrow else "wide"}"><div class="tcap">TABLE {roman(k + 1)}<br>'
            f"{TABLE_CAPTIONS[k].upper()}</div>{tb}</div>")


html_body = re.sub(r"<table>.*?</table>", wrap_table, html_body, flags=re.S)
html_body = re.sub(r"(<h2>REFERENCES</h2>)(.*)", lambda m: m.group(1) + m.group(2).replace("<p>", '<p class="ref">'), html_body, flags=re.S)

CSS = """
@page { size: A4; margin: 15mm 14mm 17mm 14mm; @bottom-center { content: counter(page); font: 8pt 'Times New Roman', serif; } }
* { box-sizing: border-box; }
body { font-family: 'Times New Roman', Times, serif; font-size: 9.6pt; line-height: 1.19; color: #000; margin: 0; }
.front { text-align: center; margin-bottom: 4mm; }
.front h1 { font-size: 20pt; font-weight: normal; line-height: 1.15; margin: 2mm 0 5mm; }
.authors { display: flex; justify-content: center; gap: 12mm; }
.authors div { font-size: 9pt; line-height: 1.25; }
.authors b { font-weight: normal; font-size: 11pt; }
.front .small { font-size: 8.5pt; margin-top: 3mm; }
.cols { column-count: 2; column-gap: 6.5mm; column-fill: balance; }
.cols p, .cols li { text-align: justify; hyphens: auto; margin: 0 0 1.6mm; orphans: 2; widows: 2; }
.abs { font-weight: bold; font-size: 9pt; line-height: 1.17; }
h2 { font-size: 9.6pt; font-weight: normal; text-transform: uppercase; text-align: center; margin: 3.5mm 0 1.4mm; break-after: avoid; letter-spacing: .02em; }
h3 { font-size: 9.6pt; font-weight: normal; font-style: italic; margin: 2.4mm 0 1mm; break-after: avoid; }
h4 { font-size: 9.6pt; font-weight: normal; font-style: italic; margin: 1.6mm 0 .6mm; break-after: avoid; }
ul, ol { margin: 0 0 1.6mm; padding-left: 5mm; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 7.6pt; overflow-wrap: anywhere; }
pre { font-family: Consolas, 'Courier New', monospace; font-size: 7.6pt; margin: 1mm 0 2mm; white-space: pre-wrap; break-inside: avoid; }
pre code { font-size: 7.6pt; }
.tbl { margin: 2mm 0 3mm; }
.tbl.wide { column-span: all; }
.tbl.narrow { break-inside: avoid; }
.tcap { font-size: 7.6pt; text-align: center; margin-bottom: 1mm; line-height: 1.15; }
table { border-collapse: collapse; width: 100%; font-size: 7.4pt; line-height: 1.13; }
th, td { border: .4pt solid #444; padding: .5mm 1.1mm; vertical-align: top; text-align: left; overflow-wrap: anywhere; }
th { font-weight: bold; background: #efefef; }
tr { break-inside: avoid; }
figure { margin: 2mm 0 3mm; text-align: center; }
figure.wide { column-span: all; break-inside: avoid; }
figure img { max-width: 118mm; max-height: 78mm; }
figcaption { font-size: 8pt; text-align: justify; margin: 1mm 4mm 0; line-height: 1.15; }
p.ref { font-size: 7.8pt; text-align: left; padding-left: 4mm; text-indent: -4mm; margin: 0 0 .9mm; hyphens: none; overflow-wrap: anywhere; }
"""
authors_html = "".join(f"<div><b>{n}</b><br>Roll no. {r}<br>CSD358 Information Retrieval<br>Shiv Nadar Institution of Eminence</div>"
                       for n, r in AUTHORS)
front = (f'<div class="front"><h1>{TITLE}</h1><div class="authors">{authors_html}</div>'
         f'<div class="small">Course project, CSD358 Information Retrieval, Monsoon 2026 (instructor: Sonia Khetarpaul). '
         f'Code, data and results: https://github.com/Deltasthicc/IRFinalProj</div></div>')
abstract_html = (f'<p class="abs"><i>Abstract</i>\u2014{inline(abstract_md)}</p>'
                 f'<p class="abs"><i>Index Terms</i>\u2014{inline(keywords_md)}.</p>')
html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{TITLE}</title><style>{CSS}</style></head>"
        f"<body>{front}<div class='cols'>{abstract_html}{html_body}</div></body></html>")
OUT.with_suffix(".html").write_text(html, encoding="utf-8")

# ---------------------------------------------------------------------------------------------------------------
# pdf
# ---------------------------------------------------------------------------------------------------------------
browser = next((p for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", shutil.which("chrome") or "",
                            shutil.which("msedge") or ""] if p and Path(p).exists()), None)
leftover = re.findall(r"@@.{0,40}|\{\{.{0,30}|Role [ABC]\b.{0,30}", md_for_file)
if leftover:
    print("WARNING leftover markers:", leftover[:10])
if browser:
    pdf = OUT.with_suffix(".pdf")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    OUT.with_suffix(".html").as_uri()], check=True, timeout=240)
    print("wrote", pdf, pdf.stat().st_size // 1024, "KB;", count, "figures,", len(tables), "tables")
else:
    print("no Chrome/Edge found; open the .html and print to PDF")
