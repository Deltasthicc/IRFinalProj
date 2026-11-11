"""Significance for system comparisons: paired bootstrap over queries, plus win/loss/tie.

With 20 test queries a plain mean difference is noisy, so every comparison reports a 95 percent
bootstrap confidence interval for the paired difference and the win/loss counts. The seed is
fixed so the numbers in the report can be regenerated exactly.
"""
from __future__ import annotations

import random
from typing import Sequence


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 13) -> dict:
    """Compare per-query scores a (system) and b (baseline), same query order."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty score lists")
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
    # two-sided bootstrap p-value for H0: mean difference = 0 (shift-to-null)
    mean_d = sum(d) / n
    centred = [x - mean_d for x in d]
    rng2 = random.Random(seed + 1)
    extreme = sum(abs(sum(centred[rng2.randrange(n)] for _ in range(n)) / n) >= abs(mean_d) - 1e-12 for _ in range(n_boot))
    return {"mean_diff": mean_d, "ci_low": lo, "ci_high": hi, "p": (extreme + 1) / (n_boot + 1),
            "wins": sum(x > 1e-9 for x in d), "losses": sum(x < -1e-9 for x in d),
            "ties": sum(abs(x) <= 1e-9 for x in d), "n": n}
