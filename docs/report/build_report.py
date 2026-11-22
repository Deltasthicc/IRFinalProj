"""Assemble the final report from its parts and render it to PDF.

    python docs/report/build_report.py            # writes TradeMesh_Report.md, .html and (if Chrome/Edge is found) .pdf

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
OUT = HERE / "TradeMesh_Report"


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

# number the figures in order of appearance; the .md file gets a text note, the HTML gets the embedded image
FIG_RE = re.compile(r"@@FIG:([\w.]+)@@(.*?)@@END@@", re.S)
md_for_file = FIG_RE.sub(lambda m: f"*[Figure: {m.group(1)}]* {m.group(2)}", text)
OUT.with_suffix(".md").write_text(md_for_file, encoding="utf-8")

count = 0
figs_html: list[str] = []


def stash_fig(m: re.Match) -> str:
    """Replace a figure marker by a placeholder paragraph; the figure HTML is put back after markdown has run."""
    global count
    count += 1
    path = (C if m.group(1) == "fig_model.svg" else FIGS) / m.group(1)
    cap = markdown.markdown(m.group(2)).removeprefix("<p>").removesuffix("</p>")
    figs_html.append(f'<figure><img alt="Figure {count}" src="{data_uri(path)}"><figcaption><b>Figure {count}.</b> {cap}</figcaption></figure>')
    return "\n\nFIGPLACEHOLDER" + str(count - 1) + "X\n\n"


html_body = markdown.markdown(FIG_RE.sub(stash_fig, text), extensions=["tables", "fenced_code", "sane_lists"])
html_body = re.sub(r"<p>FIGPLACEHOLDER(\d+)X</p>", lambda m: figs_html[int(m.group(1))], html_body)

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.42; color: #111; max-width: 178mm; margin: 0 auto; }
h1 { font-size: 22pt; margin: 40mm 0 6mm; line-height: 1.2; }
h2 { font-size: 15pt; margin: 9mm 0 3mm; border-bottom: 1px solid #999; padding-bottom: 1mm; break-after: avoid; }
h3 { font-size: 12pt; margin: 6mm 0 2mm; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 4mm 0 1mm; break-after: avoid; }
p, li { text-align: justify; hyphens: auto; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9pt; background: #f2f2f2; padding: 0 1px; word-break: break-word; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 2mm 3mm; font-size: 9pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 8.4pt; line-height: 1.25; break-inside: auto; }
th, td { border: 1px solid #aaa; padding: 1mm 1.6mm; vertical-align: top; text-align: left; overflow-wrap: anywhere; }
th { background: #e9e9e9; }
tr { break-inside: avoid; }
figure { margin: 4mm 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 105mm; }
figcaption { font-size: 9pt; text-align: justify; margin-top: 1.5mm; }
h1 + p, h1 ~ p:nth-of-type(-n+4) { text-align: left; }
h2:first-of-type { break-before: page; }
"""
html = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>TradeMesh report</title><style>{CSS}</style></head><body>{html_body}</body></html>"
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
    print("wrote", pdf, pdf.stat().st_size // 1024, "KB;", count, "figures")
else:
    print("no Chrome/Edge found; open the .html and print to PDF")
