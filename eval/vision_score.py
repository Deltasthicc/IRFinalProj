"""Step 3 of the small page-image demo: rank the pages for each question and write the results the Pages tab shows.

    python eval/vision_score.py

Reads data/vision/emb_pages.npz and emb_queries.npz (from eval/vision_encode.py), data/vision/pages.json and questions.json.
For every question it ranks all pooled pages two ways:
  * vision: our own MaxSim (trademesh.vision.maxsim) over the page-image multi-vectors;
  * text:   our own BM25 over the text that pdftotext extracted from the same pages (the text-only baseline).
It reports hit@1 and hit@3 (a hit = a gold page among the top results) and writes data/vision/results.json (the file the Pages tab
reads) and eval/out/vision_eval.json. Eight hand-written questions on 39 pages: a demonstration, not a significance test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trademesh.ranking.bm25 import BM25Index  # noqa: E402
from trademesh.types import Passage  # noqa: E402
from trademesh.vision.maxsim import rank_pages, write_results  # noqa: E402

V = ROOT / "data" / "vision"
K = 4


def main() -> None:
    pages = json.loads((V / "pages.json").read_text(encoding="utf-8"))
    questions = json.loads((V / "questions.json").read_text(encoding="utf-8"))
    ep, eq = np.load(V / "emb_pages.npz"), np.load(V / "emb_queries.npz")
    missing = [p for p in pages if p not in ep.files]
    if missing:
        sys.exit(f"embeddings missing for {len(missing)} pages: run eval/vision_encode.py without --limit")
    page_vecs = {p: ep[p] for p in pages}
    passages = [Passage(id=pid, ticker=p["ticker"], company=p["ticker"], fiscal_year=0, item="", kind="text", text=p["text"], page=p["page"])
                for pid, p in pages.items()]
    bm25 = BM25Index(passages)
    runs, rows = [], []
    for q in questions:
        vis = rank_pages(eq[q["id"]], page_vecs, k=len(pages))
        txt = [(h.passage_id, h.score) for h in bm25.search(q["question"], k=len(pages))]
        gold = set(q["gold"])
        rank = lambda ranked: next((i + 1 for i, (pid, _) in enumerate(ranked) if pid in gold), None)
        rv, rt = rank(vis), rank(txt)
        rows.append({"id": q["id"], "question": q["question"], "gold": q["gold"], "vision_rank_of_first_gold": rv, "text_rank_of_first_gold": rt,
                     "vision_top3": [p for p, _ in vis[:3]], "text_top3": [p for p, _ in txt[:3]]})
        runs.append({"query": q["question"], "gold": q["gold"], "results": [
            {"ticker": pages[pid]["ticker"], "page": pages[pid]["page"], "score": float(s), "image": pages[pid]["image"],
             "correct": pid in gold} for pid, s in vis[:K]]})
    n = len(rows)
    summ = {}
    for name, key in (("vision_maxsim", "vision_rank_of_first_gold"), ("text_bm25", "text_rank_of_first_gold")):
        ranks = [r[key] for r in rows]
        summ[name] = {"hit@1": sum(r == 1 for r in ranks), "hit@3": sum(r is not None and r <= 3 for r in ranks),
                      "mrr": round(sum(1 / r for r in ranks if r) / n, 3)}
    info = json.loads((V / "encode_info.json").read_text(encoding="utf-8")) if (V / "encode_info.json").exists() else {}
    kinds = {k: sum(1 for p in pages.values() if p["kind"] == k) for k in ("gold", "numbers", "prose")}
    write_results(V / "results.json", runs)
    res = json.loads((V / "results.json").read_text(encoding="utf-8"))
    res.update({"summary": summ, "n_questions": n, "n_pages": len(pages), "pool": kinds, "encoder": info, "note": "8 hand-written questions, 39 pages, "
                "gold = a page matching the question's keyword rule; a demonstration, not a benchmark"})
    (V / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (ROOT / "eval" / "out").mkdir(parents=True, exist_ok=True)
    (ROOT / "eval" / "out" / "vision_eval.json").write_text(json.dumps({"summary": summ, "n_questions": n, "n_pages": len(pages), "pool": kinds,
                                                                        "encoder": info, "per_question": rows}, indent=1), encoding="utf-8")
    print(f"{n} questions, {len(pages)} pages")
    for name, s in summ.items():
        print(f"  {name:14s} hit@1 {s['hit@1']}/{n}  hit@3 {s['hit@3']}/{n}  MRR {s['mrr']}")
    for r in rows:
        print(f"  {r['id']}: vision rank {r['vision_rank_of_first_gold']}, text rank {r['text_rank_of_first_gold']}")


if __name__ == "__main__":
    main()
