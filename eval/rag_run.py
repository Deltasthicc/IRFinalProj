"""Run the three RAG study configurations on the test needs and save the raw outputs.

Configurations (proposal 10.3):
  closed_book : same LLM, no passages
  bm25_rag    : BM25 context (first-stage only, no reranker, no section prior)
  best_rag    : best retriever (default: hybrid) + reranker, if B provides them

Run:  python eval/rag_run.py --split test [--bm25 NAME] [--best NAME]
Output: eval/out/rag_runs_<split>.json   (LLM calls are cached; live-call count is printed)

The test split should be run ONCE, after the feature freeze. Nothing is tuned on it.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import NEEDS, ROOT, build_pipeline, read_csv, write_json  # noqa: E402

from trademesh.rag.llm import LLMClient  # noqa: E402
from trademesh.rag.pipeline import RagConfig  # noqa: E402


def result_record(res) -> dict:
    v = res.verifier
    return {
        "mode": res.mode, "answer": res.answer, "abstained": res.abstained, "abstain_reason": res.abstain_reason,
        "confidence": res.confidence, "passage_ids": [p.id for p in res.passages], "cited": v.cited if v else None,
        "verifier_passed": v.passed if v else None, "verifier_flags": v.flags if v else None,
        "numbers_grounded": (not v.ungrounded) if v else None, "n_numbers": len(v.numbers) if v else None,
        "model": res.model, "cached": res.cached,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--bm25", default=None, help="name of the BM25 retriever")
    ap.add_argument("--best", default=None, help="name of the best (hybrid) retriever")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--allow-stub", action="store_true", help="dry run only; output is marked not reportable")
    args = ap.parse_args()

    needs = [n for n in read_csv(NEEDS / "needs.csv") if n["split"] == args.split]
    llm = LLMClient()
    cfg = RagConfig(tau=args.tau, lam=args.lam)
    bm25_pipe, bm25_name, dry1 = build_pipeline(args.bm25, args.allow_stub, False, cfg, llm)
    best_pipe, best_name, dry2 = build_pipeline(args.best, args.allow_stub, True, cfg, llm)
    dry = dry1 or dry2

    runs = []
    for n in needs:
        q = n["need"]
        runs.append({
            "id": n["id"], "question": q, "type": n["type"], "needs_table": n["needs_table"],
            "closed_book": result_record(bm25_pipe.answer(q, closed_book=True)),
            "bm25_rag": result_record(bm25_pipe.answer(q)),
            "best_rag": result_record(best_pipe.answer(q)),
        })
    out = {"split": args.split, "retrievers": {"bm25_rag": bm25_name, "best_rag": best_name},
           "model": llm.model, "tau": args.tau, "lam": args.lam, "reportable": not dry, "runs": runs}
    path = write_json(f"rag_runs_{args.split}.json", out, dry)
    print(llm.usage_line())
    print(f"wrote {path.relative_to(ROOT)} ({len(runs)} questions){'  [DRY RUN, not reportable]' if dry else ''}")


if __name__ == "__main__":
    main()
