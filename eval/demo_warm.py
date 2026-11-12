"""Warm the LLM answer cache for the live demo and show how each demo question behaves.

Runs the demo questions through the same settings as the app (hybrid retriever, cross-encoder reranker,
tau = 0, lambda = 0). Questions already in the cache cost nothing; new ones make one live call each
(about 5 seconds apart). Afterwards the demo works offline with LLM_OFFLINE=1.

Run:  python eval/demo_warm.py            (live calls allowed for uncached questions)
      LLM_OFFLINE=1 python eval/demo_warm.py   (check only: a question that is not cached stops the run)
Output: eval/out/demo_warm.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import NEEDS, build_pipeline, read_csv, write_json  # noqa: E402

from trademesh.rag.llm import CacheMiss  # noqa: E402
from trademesh.rag.pipeline import RagConfig  # noqa: E402

# (label, question or need/unanswerable id, closed_book). Ids are looked up so the text matches the study exactly.
DEMO = [
    ("1 numeric answer, cited", "R07", False),
    ("1b same question, closed-book (no passages)", "R07", True),
    ("2 typo corrected to Tesla", "How does Telsa describe the risk of depending on its Chief Executive Officer?", False),
    ("3 should decline (future year)", "U01", False),
    ("4 compare two companies", "T10", False),
    ("backup A numeric answer", "R10", False),
    ("backup A closed-book", "R10", True),
    ("backup B easy entity question", "E04", False),
    ("backup C should decline (1995)", "U09", False),
]


def resolve(ref: str) -> str:
    texts = {n["id"]: n["need"] for n in read_csv(NEEDS / "needs.csv")}
    texts.update({u["id"]: u["question"] for u in read_csv(NEEDS / "unanswerable.csv")})
    return texts.get(ref, ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--allow-stub", action="store_true", help="dry run only (not reportable)")
    args = ap.parse_args()

    cfg = RagConfig(tau=0.0, lam=0.0)
    pipe, name, dry = build_pipeline(args.retriever, args.allow_stub, not args.no_reranker, cfg)
    rows = []
    for label, ref, closed in DEMO:
        q = resolve(ref)
        try:
            res = pipe.answer(q, closed_book=closed)
        except CacheMiss:
            print(f"[NOT CACHED, offline] {label}: {q}")
            rows.append({"label": label, "question": q, "status": "not cached"})
            continue
        v = res.verifier
        status = ("declined" if res.abstained else res.mode)
        rows.append({"label": label, "question": q, "closed_book": closed, "status": status, "cached": res.cached,
                     "corrections": res.corrections, "companies": list(res.by_company) or [m.ticker for m in res.mentions],
                     "verifier_passed": None if v is None else v.passed,
                     "verifier_flags": [] if v is None else v.flags, "answer": res.answer})
        print(f"\n[{label}] {q}\n  mode={res.mode} status={status} source={'cache' if res.cached else 'LIVE call'} "
              f"verifier={'n/a' if v is None else ('pass' if v.passed else 'FLAGGED ' + '; '.join(v.flags))}"
              + (f" corrections={res.corrections}" if res.corrections else ""))
        print("  " + (res.answer or "").replace("\n", "\n  ")[:600])
    print("\n" + pipe.llm.usage_line())
    print(f"wrote {write_json('demo_warm.json', {'retriever': name, 'rows': rows}, dry)}"
          + ("  [DRY RUN]" if dry else ""))


if __name__ == "__main__":
    main()
