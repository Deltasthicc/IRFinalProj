"""Full-corpus page-image run, step 3: rank every page for every question with our MaxSim and with BM25 over the page text.

    python eval/vision_full_score.py --work /content/drive/MyDrive/TradeMesh_vision [--out data/vision_full]

Reads <work>/pages.json, questions.json, emb/<page>.npy and emb_queries.npz. For each question, all pages of all 20 filings compete.
  * vision: MaxSim (trademesh.vision.maxsim.maxsim) between the question vectors and each page's image vectors;
  * text:   BM25 (trademesh.ranking.bm25) over the text extracted from the same pages, with the company name added to each page.
Reports hit@1, hit@3, hit@10 and MRR overall and per question group, and writes
  <out>/results.json   the file the Pages tab shows (questions, best pages, summary), with small JPEG copies of the best pages in <out>/thumbs/
  <out>/eval.json      per-question ranks (also copied to eval/out/vision_full_eval.json)
A hit means a gold page (a page matching the question's keyword rule) is among the top results. The keyword rule is text based, which
favours the text baseline a little; the questions of the "hand-written" group were written by us. Read the numbers as a larger demonstration
of the idea on the real corpus, not as a benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trademesh.ranking.bm25 import BM25Index  # noqa: E402
from trademesh.types import Passage  # noqa: E402
from trademesh.vision.maxsim import maxsim  # noqa: E402

K_SHOW = 4


def maxsim_matrix(q: np.ndarray, page_vecs: list[np.ndarray]) -> np.ndarray:
    """MaxSim of one question (n, d) against every page (m_i, d), using the project's own maxsim()."""
    q32 = q.astype(np.float32)
    return np.array([maxsim(q32, p.astype(np.float32)) for p in page_vecs])


def ranks_of_first_gold(order: list[str], gold: set[str]) -> int | None:
    return next((i + 1 for i, pid in enumerate(order) if pid in gold), None)


def summarise(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {"hit@1": sum(r == 1 for r in ranks), "hit@3": sum(r is not None and r <= 3 for r in ranks),
            "hit@10": sum(r is not None and r <= 10 for r in ranks), "mrr": round(sum(1 / r for r in ranks if r) / n, 3) if n else 0.0, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "vision_full"))
    args = ap.parse_args()
    work, out = Path(args.work).resolve(), Path(args.out).resolve()
    from PIL import Image
    pages = json.loads((work / "pages.json").read_text(encoding="utf-8"))
    questions = json.loads((work / "questions.json").read_text(encoding="utf-8"))
    companies = {c["ticker"]: c["name"] for c in json.loads((ROOT / "data" / "companies.json").read_text(encoding="utf-8"))}
    ids = [pid for pid in pages if (work / "emb" / f"{pid}.npy").exists()]
    if len(ids) < len(pages):
        print(f"warning: {len(pages) - len(ids)} of {len(pages)} pages have no vectors and are left out", file=sys.stderr)
    questions = [q for q in questions if any(g in set(ids) for g in q["gold"])]
    qv = np.load(work / "emb_queries.npz")
    page_vecs = [np.load(work / "emb" / f"{pid}.npy") for pid in ids]
    bm25 = BM25Index([Passage(id=pid, ticker=pages[pid]["ticker"], company=companies.get(pages[pid]["ticker"], pages[pid]["ticker"]), fiscal_year=0,
                              item="", kind="text", text=pages[pid]["text"], page=pages[pid]["page"]) for pid in ids])
    rows, runs = [], []
    for q in questions:
        gold = set(q["gold"]) & set(ids)
        vis_scores = maxsim_matrix(qv[q["id"]], page_vecs)
        order = np.argsort(-vis_scores, kind="stable")
        vis = [(ids[i], float(vis_scores[i])) for i in order]
        txt = [(h.passage_id, h.score) for h in bm25.search(q["question"], k=len(ids))]
        rv, rt = ranks_of_first_gold([p for p, _ in vis], gold), ranks_of_first_gold([p for p, _ in txt], gold)
        rows.append({"id": q["id"], "group": q["group"], "question": q["question"], "company": q["company"], "gold": sorted(gold),
                     "vision_rank": rv, "text_rank": rt, "vision_top3": [p for p, _ in vis[:3]], "text_top3": [p for p, _ in txt[:3]]})
        runs.append({"query": q["question"], "id": q["id"], "group": q["group"], "gold": sorted(gold), "results": [
            {"ticker": pages[pid]["ticker"], "page": pages[pid]["page"], "score": s, "image": f"data/vision_full/thumbs/{pid}.jpg", "correct": pid in gold}
            for pid, s in vis[:K_SHOW]]})
    thumbs = out / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    for run in runs:
        for r in run["results"]:
            dst = thumbs / f"{r['ticker']}_p{r['page']}.jpg"
            if not dst.exists():
                Image.open(work / pages[f"{r['ticker']}_p{r['page']}"]["image"]).convert("RGB").save(dst, quality=72, optimize=True)
    groups = sorted({r["group"] for r in rows})
    summ = {"overall": {"vision_maxsim": summarise([r["vision_rank"] for r in rows]), "text_bm25": summarise([r["text_rank"] for r in rows])}, "groups": {}}
    for g in groups:
        sub = [r for r in rows if r["group"] == g]
        summ["groups"][g] = {"vision_maxsim": summarise([r["vision_rank"] for r in sub]), "text_bm25": summarise([r["text_rank"] for r in sub])}
    info = json.loads((work / "encode_info.json").read_text(encoding="utf-8")) if (work / "encode_info.json").exists() else {}
    result = {"queries": runs, "summary": summ, "n_questions": len(rows), "n_pages": len(ids), "n_companies": len({pages[p]["ticker"] for p in ids}),
              "encoder": info, "scale": "full",
              "note": "Full corpus: every page of the latest 10-K of each company competes for every question. Gold pages come from a keyword rule on the page text, "
                      "which favours the text baseline; a demonstration of the idea, not a benchmark."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    ev = {"summary": summ, "n_questions": len(rows), "n_pages": len(ids), "encoder": info, "per_question": rows}
    (out / "eval.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
    (ROOT / "eval" / "out").mkdir(parents=True, exist_ok=True)
    (ROOT / "eval" / "out" / "vision_full_eval.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
    print(f"{len(rows)} questions over {len(ids)} pages of {result['n_companies']} companies")
    for name in ("vision_maxsim", "text_bm25"):
        s = summ["overall"][name]
        print(f"  {name:14s} hit@1 {s['hit@1']}/{s['n']}  hit@3 {s['hit@3']}/{s['n']}  hit@10 {s['hit@10']}/{s['n']}  MRR {s['mrr']}")
    for g in groups:
        v, t = summ["groups"][g]["vision_maxsim"], summ["groups"][g]["text_bm25"]
        print(f"  {g:17s} vision hit@1 {v['hit@1']}/{v['n']}  text hit@1 {t['hit@1']}/{t['n']}")


if __name__ == "__main__":
    main()
