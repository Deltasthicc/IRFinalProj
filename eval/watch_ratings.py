"""Watch the remote for another rater's finished human ratings, import them, and re-score.

    python eval/watch_ratings.py --once                 # look now; exit code 0 = nothing new, 10 = a new rater was imported
    python eval/watch_ratings.py --interval 300         # keep looking every 5 minutes until one arrives
    python eval/watch_ratings.py --once --no-import     # only report what the remote holds

What it does, in order:
 1. `git fetch --prune <remote>` (it reads every remote branch, so it does not matter which branch the rater pushes to).
 2. For every rater folder `eval/human/raters/<name>/` that exists on the remote but is not here yet, it reads
    references.csv, ratings.csv and citations.csv straight from git (nothing from the remote is executed or checked out)
    and validates them against eval/human/blind_answers.json: every (question, label) rated 0, 1 or 2, every citation
    marked 0 or 1, one reference per question. An incomplete or invalid sheet is reported and NOT imported.
 3. A complete sheet is written to eval/human/raters/<name>/ (plus judgments/pool_sheet_<name>.csv if it was pushed), then
    `python eval/human_score.py` is run over all raters, which rewrites eval/out/rag_scores_human.json and
    docs/report_C/human_rating_results.md. The app's Evaluation tab reads that JSON, so it updates by itself.
 4. It prints the files whose wording depends on the number of raters. Those sentences (report, README, demo script) are
    written by hand from the new numbers, because they say who rated and what the agreement was; this script never writes
    prose and never invents a rating. Nothing is committed or pushed.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATERS = Path("eval/human/raters")
JUDG = Path("eval/human/judgments")
NAME_OK = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,30}$")
NEW_RATER_EXIT = 10


def git(repo: Path, *args: str, check: bool = True) -> bytes:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if r.returncode == 0 else b""


def _rows(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def expected(repo: Path) -> dict:
    """What a complete sheet must contain, from the committed blind package."""
    blind = json.loads((repo / "eval/human/blind_answers.json").read_text(encoding="utf-8"))
    qids = [q["id"] for q in blind["questions"]]
    answers = {(q["id"], lab) for q in blind["questions"] for lab in q["answers"]}
    cites = {(q["id"], lab, c["ref"]) for q in blind["questions"] for lab, a in q["answers"].items() for c in a["cited"]}
    return {"qids": qids, "answers": answers, "cites": cites}


def validate(files: dict[str, bytes], exp: dict, allow_partial_citations: bool = False) -> list[str]:
    """Problems with a rater's three sheets (an empty list means complete and valid).

    With allow_partial_citations the citation sheet may be incomplete (every filled mark must still be valid and refer to a real
    citation); the ratings and references must always be complete, because they carry the headline result."""
    problems = []
    for need in ("references.csv", "ratings.csv", "citations.csv"):
        if need not in files:
            problems.append(f"{need} is missing")
    if problems:
        return problems
    refs = {r["question_id"]: r for r in _rows(files["references.csv"])}
    missing = [q for q in exp["qids"] if q not in refs or not refs[q].get("reference", "").strip()]
    if missing:
        problems.append(f"{len(missing)} questions have no reference answer")
    got = {}
    for r in _rows(files["ratings.csv"]):
        if r["rating"].strip() != "":
            if r["rating"].strip() not in ("0", "1", "2"):
                problems.append(f"rating {r['rating']!r} for {r['question_id']}/{r['label']} is not 0, 1 or 2")
            got[(r["question_id"], r["label"])] = r["rating"]
    if set(got) != exp["answers"]:
        problems.append(f"{len(got)} of {len(exp['answers'])} answers rated")
    cites = {}
    for r in _rows(files["citations.csv"]):
        if r["supports"].strip() != "":
            if r["supports"].strip() not in ("0", "1"):
                problems.append(f"citation mark {r['supports']!r} is not 0 or 1")
            cites[(r["question_id"], r["label"], r["citation"])] = r["supports"]
    if allow_partial_citations:
        stray = set(cites) - exp["cites"]
        if stray:
            problems.append(f"{len(stray)} citation marks do not match the blind package")
    elif set(cites) != exp["cites"]:
        problems.append(f"{len(cites)} of {len(exp['cites'])} citations checked")
    return problems


def remote_raters(repo: Path, remote: str) -> dict[str, tuple[str, dict[str, bytes]]]:
    """{rater name: (ref, {file name: bytes})} for the best-filled copy of each rater folder on any remote branch."""
    refs = [r for r in git(repo, "for-each-ref", "--format=%(refname:short)", f"refs/remotes/{remote}").decode().split()
            if not r.endswith("/HEAD")]
    best: dict[str, tuple[int, str, dict[str, bytes]]] = {}
    for ref in refs:
        paths = git(repo, "ls-tree", "-r", "--name-only", ref, "--", RATERS.as_posix(), JUDG.as_posix(), check=False).decode().split("\n")
        names = {p.split("/")[3] for p in paths if p.startswith(RATERS.as_posix() + "/") and p.count("/") == 4}
        for name in names:
            if not NAME_OK.match(name):
                print(f"ignoring rater folder with an unsafe name on {ref}: {name!r}")
                continue
            files = {}
            for fn in ("references.csv", "ratings.csv", "citations.csv"):
                blob = git(repo, "show", f"{ref}:{RATERS.as_posix()}/{name}/{fn}", check=False)
                if blob:
                    files[fn] = blob
            sheet = git(repo, "show", f"{ref}:{JUDG.as_posix()}/pool_sheet_{name}.csv", check=False)
            if sheet:
                files["pool_sheet.csv"] = sheet
            filled = sum(1 for r in _rows(files.get("ratings.csv", b"question_id,label,rating\n")) if r.get("rating", "").strip() != "")
            if name not in best or filled > best[name][0]:
                best[name] = (filled, ref, files)
    return {n: (ref, files) for n, (_, ref, files) in best.items()}


def import_dir(repo: Path, src: Path, name: str, allow_partial_citations: bool = False) -> list[str]:
    """Import one rater from an unpacked folder (for example a ZIP the rater sent) after the same validation as a remote
    sheet. `src` is the folder that holds references.csv, ratings.csv and citations.csv. Returns the problems (empty = imported).
    Only those three CSV files are read; nothing else in the folder is used."""
    if not NAME_OK.match(name):
        return [f"unsafe rater name {name!r}"]
    if (repo / RATERS / name).exists():
        return [f"{name} already exists here"]
    files = {fn: (src / fn).read_bytes() for fn in ("references.csv", "ratings.csv", "citations.csv") if (src / fn).is_file()}
    problems = validate(files, expected(repo), allow_partial_citations)
    if not problems:
        rdir = repo / RATERS / name
        rdir.mkdir(parents=True)
        for fn, data in files.items():
            (rdir / fn).write_bytes(data)
    return problems


def check(repo: Path, remote: str = "origin", do_import: bool = True, fetch: bool = True) -> dict:
    """One look at the remote. Returns {'imported': [...], 'incomplete': {name: problems}, 'known': [...]}."""
    if fetch:
        git(repo, "fetch", "--prune", remote)
    exp = expected(repo)
    known = sorted(p.name.lower() for p in (repo / RATERS).glob("*") if p.is_dir())
    result = {"imported": [], "incomplete": {}, "known": known}
    for name, (ref, files) in sorted(remote_raters(repo, remote).items()):
        if name.lower() in known:
            continue
        problems = validate(files, exp)
        if problems:
            result["incomplete"][name] = {"ref": ref, "problems": problems}
            continue
        if do_import:
            rdir = repo / RATERS / name
            rdir.mkdir(parents=True)
            for fn in ("references.csv", "ratings.csv", "citations.csv"):
                (rdir / fn).write_bytes(files[fn])
            if "pool_sheet.csv" in files:
                (repo / JUDG).mkdir(parents=True, exist_ok=True)
                (repo / JUDG / f"pool_sheet_{name}.csv").write_bytes(files["pool_sheet.csv"])
        result["imported"].append({"name": name, "ref": ref})
    return result


FOLLOW_UP = [
    "docs/report/parts/*.md and docs/report_B, docs/report_C drafts: every sentence that says one rater, one human or a single project member",
    "README.md and docs/demo_script.md: the headline answer ratings and the one-rater limitation",
    "src/trademesh/app/backend.py and the Evaluation tab text: counts of raters",
    "docs/report/figures (python docs/report/figures/make_figures.py) and python docs/report/build_report.py",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--once", action="store_true", help="check once and exit")
    ap.add_argument("--interval", type=int, default=300, help="seconds between checks when looping")
    ap.add_argument("--no-import", action="store_true", help="only report, do not write files or score")
    ap.add_argument("--from-dir", help="import a rater from an unpacked folder holding the three CSVs (needs --name) instead of a remote")
    ap.add_argument("--name", help="rater name for --from-dir")
    ap.add_argument("--allow-partial-citations", action="store_true",
                    help="with --from-dir: accept an incomplete citation sheet (ratings and references must be complete)")
    args = ap.parse_args()
    if args.from_dir:
        if not args.name:
            sys.exit("--from-dir needs --name")
        problems = import_dir(ROOT, Path(args.from_dir), args.name, args.allow_partial_citations)
        if problems:
            sys.exit(f"{args.name} not imported: " + "; ".join(problems))
        print(f"imported {args.name} from {args.from_dir}")
        subprocess.run([sys.executable, str(ROOT / "eval" / "human_score.py")], check=True, cwd=ROOT)
        return NEW_RATER_EXIT
    while True:
        res = check(ROOT, args.remote, do_import=not args.no_import)
        stamp = time.strftime("%H:%M:%S")
        for name, info in res["incomplete"].items():
            print(f"[{stamp}] {name} on {info['ref']} is not ready: " + "; ".join(info["problems"]))
        if res["imported"]:
            for r in res["imported"]:
                print(f"[{stamp}] NEW RATER {r['name']} (from {r['ref']}): complete" + ("" if args.no_import else ", imported"))
            if not args.no_import:
                subprocess.run([sys.executable, str(ROOT / "eval" / "human_score.py")], check=True, cwd=ROOT)
                print("\nScores and docs/report_C/human_rating_results.md are updated. Still to do by hand (the wording depends on who rated):")
                for line in FOLLOW_UP:
                    print("  -", line)
            return NEW_RATER_EXIT
        if not res["incomplete"]:
            print(f"[{stamp}] nothing new (raters here: {', '.join(res['known']) or 'none'})")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
