"""Is the pooled test collection fair to every system? (Zobel 1998; Buckley and Voorhees 2004)

Leave-one-system-out: remove from the judgments the passages that ONLY system S put into the pool
(those it alone "discovered"), re-score all systems, and report how S's score and the ordering of
all systems change. A big drop for S means S owes its score to the pool it helped build; a stable
ordering (Kendall tau near 1) means conclusions do not depend on any one system. Also prints judged@10.
Needs data/judgments/qrels.csv and pool_key.json (from eval/make_pools.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_common import JUDG, load_needs, mean, require_qrels, run_system, setup, table, write_json  # noqa: E402

from trademesh.evaluation.metrics import average_precision, kendall_tau, ndcg_at_k  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-stub", action="store_true")
    args = ap.parse_args()
    key_path = JUDG / "pool_key.json"
    if not key_path.exists():
        sys.exit("TODO(run): data/judgments/pool_key.json is missing; run eval/make_pools.py first.")
    key = json.loads(key_path.read_text())
    passages, retrievers, _, dry = setup(args.allow_stub)
    needs = load_needs("all")
    qrels = require_qrels(needs, dry)
    by_id = {p.id: p for p in passages}
    runs = {s: run_system(r.search, needs, by_id, k=30) for s, r in retrievers.items()}
    systems = list(runs)

    def score(sys_name: str, drop_unique_of: str | None) -> tuple[float, float]:
        ap_, nd = [], []
        for n in needs:
            q = n["id"]
            rel = qrels[q]
            if drop_unique_of:
                contrib = key[q]["contributors"]
                rel = {d: g for d, g in rel.items() if contrib.get(d) != [drop_unique_of]}
            ap_.append(average_precision(runs[sys_name][q], rel))
            nd.append(ndcg_at_k(runs[sys_name][q], rel, 10, binary=True))
        return mean(ap_), mean(nd)

    full = {s: score(s, None) for s in systems}
    rows, out = [], {}
    for s in systems:
        loo = {t: score(t, s) for t in systems}
        tau = kendall_tau([full[t][0] for t in systems], [loo[t][0] for t in systems])
        rows.append([s, f"{full[s][0]:.3f}", f"{loo[s][0]:.3f}", f"{loo[s][0] - full[s][0]:+.3f}", f"{tau:.2f}"])
        out[s] = {"MAP_full": full[s][0], "MAP_without_own_unique": loo[s][0], "tau_system_order": tau}
    print("Leave-one-system-out (MAP):" + ("   [DRY RUN]" if dry else ""))
    print(table(["system", "MAP", "MAP w/o own unique", "change", "tau of system order"], rows))
    print(f"\nwrote {write_json('pool_bias.json', out, dry)}")


if __name__ == "__main__":
    main()
