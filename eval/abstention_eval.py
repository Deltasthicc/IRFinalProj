"""Abstention evaluation (proposal 10.3): positive class = "should abstain".

Items: the answerable needs (should NOT abstain) and the unanswerable questions (should abstain).
tau is tuned on the DEV items only (10 dev needs + dev unanswerable) and then frozen. The headline
numbers are on the TEST items (20 test needs + test unanswerable). We also print all items combined,
because the proposal asks for 20 + 10, clearly marked as partly tuned-on.

The system abstains if confidence < tau OR the model answers INSUFFICIENT EVIDENCE.
--score-only evaluates the score rule alone and makes no LLM calls.

Run:  python eval/abstention_eval.py [--score-only] [--retriever NAME] [--reranker]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import NEEDS, ROOT, build_pipeline, read_csv, write_json  # noqa: E402

from trademesh.rag.abstain import abstention_metrics, tune_tau  # noqa: E402
from trademesh.rag.llm import LLMClient  # noqa: E402
from trademesh.rag.pipeline import RagConfig  # noqa: E402


def load_items() -> list[dict]:
    items = [{"id": n["id"], "split": n["split"], "question": n["need"], "should_abstain": False}
             for n in read_csv(NEEDS / "needs.csv")]
    items += [{"id": u["id"], "split": u["split"], "question": u["question"], "should_abstain": True}
              for u in read_csv(NEEDS / "unanswerable.csv")]
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--retriever", default=None)
    ap.add_argument("--reranker", action="store_true")
    ap.add_argument("--allow-stub", action="store_true", help="dry run only; output is marked not reportable")
    args = ap.parse_args()

    # tau = 0 and call_llm_when_low_score: we want the raw confidence and the model's own verdict
    cfg = RagConfig(tau=0.0, call_llm_when_low_score=True)
    llm = LLMClient(provider="stub") if args.score_only else LLMClient()
    pipe, name, dry = build_pipeline(args.retriever, args.allow_stub, args.reranker, cfg, llm,
                                     needs_llm=not args.score_only)

    items = load_items()
    for it in items:
        res = pipe.prepare(it["question"])
        it["confidence"] = res.confidence
        it["model_said_insufficient"] = False
        if not args.score_only and res.passages:
            full = pipe.answer(it["question"])
            it["model_said_insufficient"] = full.model_said_insufficient

    dev = [i for i in items if i["split"] == "dev"]
    test = [i for i in items if i["split"] == "test"]
    tau, dev_m = tune_tau([(i["confidence"], i["should_abstain"], i["model_said_insufficient"]) for i in dev])

    def score(subset):
        return abstention_metrics([(i["should_abstain"], i["confidence"] < tau or i["model_said_insufficient"])
                                   for i in subset])

    test_m, all_m = score(test), score(items)
    out = {"retriever": name, "score_only": args.score_only, "tau": tau, "reportable": not dry,
           "dev_tuned": dev_m.__dict__, "test": test_m.__dict__, "all_items": all_m.__dict__,
           "n": {"dev": len(dev), "test": len(test)}, "items": items}
    path = write_json("abstention_eval.json", out, dry)
    print(llm.usage_line())
    print(f"tau (tuned on dev) = {tau:.4f}")
    for label, m in (("dev (tuned on)", dev_m), ("TEST (headline)", test_m), ("all items (partly tuned-on)", all_m)):
        print(f"{label:28s} P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} acc={m.accuracy:.3f}  "
              f"tp={m.tp} fp={m.fp} fn={m.fn} tn={m.tn}")
    print(f"wrote {path.relative_to(ROOT)}{'  [DRY RUN, not reportable]' if dry else ''}")


if __name__ == "__main__":
    main()
