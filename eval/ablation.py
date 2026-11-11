"""Ablations (RQ2). Each row changes ONE thing from the default; a result is a number with a CI, not an opinion.

    python eval/ablation.py --split dev      # exploration
    python eval/ablation.py --split test     # once, after the freeze (locked)

  A  preprocessing grid for BM25: stop words x stemming x numbers x metadata prefix (16 configs)
  B  structure: drop table chunks; entity filter off
  C  fusion: RRF k in {10, 60, 100}, CombSUM, CombMNZ, (needs dense)
  D  pseudo-relevance feedback: k_docs x expansion terms
Outputs mean AP and nDCG@10 with the paired-bootstrap CI against the default configuration.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_common import claim_test_run, load_needs, mean, require_qrels, run_system, setup, table, write_json  # noqa: E402

from trademesh.evaluation.metrics import average_precision, ndcg_at_k  # noqa: E402
from trademesh.evaluation.stats import paired_bootstrap  # noqa: E402
from trademesh.ranking import (Analyzer, AnalyzerConfig, BM25Index, HybridRetriever, PRFRetriever,  # noqa: E402
                               DEFAULT_CONFIG)
from trademesh.ranking.bm25 import BM25Index as _B  # noqa: E402,F401
from trademesh.ranking import _params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-stub", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    passages, retrievers, _, dry = setup(args.allow_stub)
    needs = load_needs(args.split)
    if args.split == "test":
        claim_test_run("ablation", dry, args.force)
    qrels = require_qrels(needs, dry)
    by_id = {p.id: p for p in passages}
    k1, b = _params()
    ids = [n["id"] for n in needs]

    def evaluate(search, **kw):
        runs = run_system(search, needs, by_id, k=30, **kw)
        return ([ndcg_at_k(runs[q], qrels[q], 10, binary=True) for q in ids], [average_precision(runs[q], qrels[q]) for q in ids])

    base_ix = BM25Index(passages, Analyzer(DEFAULT_CONFIG), k1=k1, b=b)
    base_ndcg, base_ap = evaluate(base_ix.search)
    rows, record = [], {}

    def add(group: str, label: str, search, **kw):
        nd, ap_ = evaluate(search, **kw)
        s = paired_bootstrap(nd, base_ndcg)
        rows.append([group, label, f"{mean(ap_):.3f}", f"{mean(nd):.3f}", f"{s['mean_diff']:+.3f}",
                     f"[{s['ci_low']:+.3f},{s['ci_high']:+.3f}]", f"{s['wins']}/{s['losses']}/{s['ties']}"])
        record[f"{group}:{label}"] = {"AP": mean(ap_), "nDCG@10": mean(nd), "vs_default": s}

    add("default", DEFAULT_CONFIG.label(), base_ix.search)
    for stop, stem, num, meta in itertools.product([True, False], [False, True], [True, False], [False, True]):
        cfg = AnalyzerConfig(stop, stem, num, meta)
        if cfg == DEFAULT_CONFIG:
            continue
        add("A prep", cfg.label(), BM25Index(passages, Analyzer(cfg), k1=k1, b=b).search)
    add("B struct", "drop table chunks", base_ix.search, drop_tables=True)
    add("B struct", "entity filter off", base_ix.search, entity_filter=False)
    if "dense" in retrievers:
        for kk in (10, 60, 100):
            add("C fusion", f"rrf k={kk}", HybridRetriever([base_ix, retrievers["dense"]], rrf_k=kk).search)
        for m in ("combsum", "combmnz"):
            add("C fusion", m, HybridRetriever([base_ix, retrievers["dense"]], method=m).search)
    for kd, mt in itertools.product((3, 5, 10), (5, 10, 20)):
        add("D prf", f"k={kd}, m={mt}", PRFRetriever(base_ix, k_docs=kd, m_terms=mt).search)

    print(f"split={args.split} needs={len(needs)} default AP={mean(base_ap):.3f} nDCG@10={mean(base_ndcg):.3f}"
          + ("   [DRY RUN]" if dry else ""))
    print(table(["group", "config", "AP", "nDCG@10", "diff", "95% CI", "W/L/T"], rows))
    print(f"\nwrote {write_json(f'ablation_{args.split}.json', record, dry)}")


if __name__ == "__main__":
    main()
