"""Pooled test-collection construction (proposal 10.2).

For each need, the top `depth` passages of every system are merged (duplicates removed), shuffled
with a fixed seed, and written to a judging sheet that shows NO system names and NO ranks. The
mapping back to systems is written to a separate `pool_key.json` that judges must not open.
A system's contribution to the pool is recorded so that `eval/pool_bias.py` can ask what would
have happened had that system not been pooled (Zobel 1998).
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence

from ..types import Hit


def build_pool(runs: Mapping[str, Sequence[Hit]], depth: int = 8, seed: int = 7) -> tuple[list[str], dict[str, list[str]]]:
    """runs: {system: ranked hits} for ONE query. Returns (shuffled passage ids, {id: [systems that put it in the top depth]})."""
    contrib: dict[str, list[str]] = {}
    for system in sorted(runs):
        for h in runs[system][:depth]:
            contrib.setdefault(h.passage_id, []).append(system)
    ids = sorted(contrib)
    random.Random(seed).shuffle(ids)
    return ids, contrib


def assign_judges(query_ids: Sequence[str], judges: Sequence[str], n_double: int = 10, seed: int = 11) -> dict[str, list[str]]:
    """Round-robin first judge for every query; the `n_double` queries (spread over types by the
    caller's ordering) additionally get a second judge, for the kappa estimate."""
    out = {q: [judges[i % len(judges)]] for i, q in enumerate(query_ids)}
    rng = random.Random(seed)
    for q in rng.sample(list(query_ids), min(n_double, len(query_ids))):
        others = [j for j in judges if j not in out[q]]
        out[q].append(rng.choice(others))
    return out
