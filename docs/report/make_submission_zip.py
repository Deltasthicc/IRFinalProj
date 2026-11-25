"""Build the submission ZIP from the last commit plus the built report.

    python docs/report/make_submission_zip.py                       # -> ../CSD358_ShashwatRajan_TanishShitanshu_RuchirSharma.zip
    python docs/report/make_submission_zip.py --out C:\\path\\name.zip

The instructor's template is CSD358_GroupMember1_Member2_Member3; the team chose the three full names as the ZIP name.
Contents: the report PDF, GITHUB_LINK.txt, the requirements traceability table, `code/` (git archive of HEAD, so only committed
files, never .env or caches), and `dataset/` (processed passages, needs, judgments, graph, results, human ratings).
Commit first: the code folder is exactly HEAD. The script refuses to run if the report PDF is older than the report sources.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "CSD358_ShashwatRajan_TanishShitanshu_RuchirSharma"
README = """TradeMesh: Search and Verified Question Answering over SEC 10-K Filings
CSD358 Information Retrieval, Monsoon 2026
Shashwat Rajan (2510111019), Tanish Shitanshu (2510110008), Ruchir Sharma (2310110728)

Contents
  TradeMesh_Report.pdf             the report: a short IEEE-style paper with the instructor's sections, in her order
  GITHUB_LINK.txt                  the repository
  REQUIREMENTS_TRACEABILITY.md     every requested item: where it is, or why it is not
  extra/TradeMesh_Report_Detailed.pdf   the long version with every table and ablation (optional reading)
  code/                            full source (python package src/trademesh, tests, eval scripts, app, Dockerfile, report build)
  dataset/data/                    processed passages (passages.jsonl, 11,964), companies, filings list,
                                   information needs (needs/needs.csv, unanswerable.csv, typos.csv),
                                   relevance judgments (judgments/), company graph, retrieval results
  dataset/human_ratings/           the blind human ratings of the RAG answers (one folder per rater) and the blind answer file
  dataset/results/                 every result file behind the report's numbers (eval/out)

Run (details in code/README.md)
  cd code
  pip install -r requirements.txt            (add requirements-ml.txt for the dense retriever and reranker)
  python -m pytest -q                        (offline)
  python -m streamlit run src/trademesh/app/streamlit_app.py
  or: docker compose up --build

The dataset used by every experiment is the processed corpus in dataset/data/passages.jsonl (the latest 10-K of 20 companies split
into 11,964 passages, with filings.json listing each filing's accession number and URL), together with the questions, relevance
judgments, ratings and results listed above. The original SEC filing files are public and are not bundled;
python -m trademesh.ingest.corpus downloads them again from SEC EDGAR (needs SEC_USER_AGENT with a contact email).
No API key is included. Live answers need GEMINI_API_KEY in a local .env; cached demo answers work offline (LLM_OFFLINE=1).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT.parent / f"{NAME}.zip"))
    ap.add_argument("--report-pdf", help="package this PDF as TradeMesh_Report.pdf (for a trial build from a draft); the freshness check is skipped")
    args = ap.parse_args()
    pdf = Path(args.report_pdf) if args.report_pdf else ROOT / "docs/report/TradeMesh_Report.pdf"
    detailed = ROOT / "docs/report/TradeMesh_Report_Detailed.pdf"
    short_sources = [ROOT / "docs/report/concise/paper.md", ROOT / "docs/report/build_concise.py", *(ROOT / "eval/out").glob("*.json")]
    long_sources = [*(ROOT / "docs/report/parts").glob("*.md"), *(ROOT / "docs/report_B").glob("*.md"), *(ROOT / "docs/report_C").glob("*.md")]
    for f, srcs, cmd in ((pdf, short_sources, "build_concise.py"), (detailed, long_sources, "build_report.py")):
        if f is pdf and args.report_pdf:
            if not f.exists():
                sys.exit(f"{f} does not exist")
            continue
        if not f.exists() or any(s.stat().st_mtime > f.stat().st_mtime + 1 for s in srcs):
            sys.exit(f"{f.name} is missing or older than its sources: run python docs/report/{cmd} first.")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("Uncommitted changes (the code folder is exactly HEAD). Commit first:\n" + dirty)
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD"], capture_output=True, check=True).stdout
    out = Path(args.out)
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z, tarfile.open(fileobj=BytesIO(archive)) as tar:
        for m in tar.getmembers():
            if m.isfile():
                z.writestr(f"{NAME}/code/{m.name}", tar.extractfile(m).read())

        def add(src: Path, dest: str) -> None:
            if src.is_file():
                z.write(src, f"{NAME}/{dest}")

        add(pdf, "TradeMesh_Report.pdf")
        add(detailed, "extra/TradeMesh_Report_Detailed.pdf")
        add(ROOT / "docs/report/REQUIREMENTS_TRACEABILITY.md", "REQUIREMENTS_TRACEABILITY.md")
        z.writestr(f"{NAME}/GITHUB_LINK.txt", "Code, data and results: https://github.com/Deltasthicc/IRFinalProj\n")
        z.writestr(f"{NAME}/README.txt", README)
        for p in (ROOT / "data").rglob("*"):
            rel = p.relative_to(ROOT / "data")
            heavy = rel.parts[0] == "vision" and (len(rel.parts) > 1 and rel.parts[1] in ("html", "pdf", "wide") or rel.name.startswith("emb_"))
            if p.is_file() and rel.parts[0] not in ("raw", "raw_html", "vision_full_work") and not heavy:
                add(p, f"dataset/data/{rel.as_posix()}")
        for p in (ROOT / "eval/human/raters").rglob("*"):
            add(p, f"dataset/human_ratings/raters/{p.relative_to(ROOT / 'eval/human/raters').as_posix()}")
        add(ROOT / "eval/human/blind_answers.json", "dataset/human_ratings/blind_answers.json")
        for p in (ROOT / "eval/out").glob("*"):
            if p.suffix in (".json", ".csv"):
                add(p, f"dataset/results/{p.name}")
    names = zipfile.ZipFile(out).namelist()
    need = ["TradeMesh_Report.pdf", "GITHUB_LINK.txt", "README.txt", "REQUIREMENTS_TRACEABILITY.md", "extra/TradeMesh_Report_Detailed.pdf",
            "code/README.md", "code/requirements.txt", "dataset/data/passages.jsonl", "dataset/data/filings.json",
            "dataset/human_ratings/raters/Shashwat/ratings.csv", "dataset/human_ratings/raters/Ruchir/ratings.csv", "dataset/results/rag_scores_human.json"]
    missing = [n for n in need if f"{NAME}/{n}" not in names]
    stale = [n for n in names if any(s in n for s in ("_Draft", "_IEEE"))]
    if missing or stale:
        out.unlink(missing_ok=True)
        sys.exit(f"ZIP check failed. Missing: {missing}. Stale draft or old-name files: {stale}")
    bad = [n for n in names if any(s in n.lower() for s in (".env\0", "secret")) or n.endswith("/.env")]
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(names)} files); suspicious names: {bad or 'none'}")


if __name__ == "__main__":
    main()
