"""Wildcard term expansion with a permuterm index and a k-gram index (Role A, proposal 9.1).

Permuterm: every rotation of term+"$" is stored in a sorted list. For X*Y we rotate the query to Y$X and
do a prefix search with binary search. This handles exactly one '*'.
k-gram (k = 2): map each character bigram of "$term$" to the terms containing it; a pattern is answered
by intersecting the lists of its bigrams, then post-filtering with a regular expression (bigrams alone
admit false positives). This handles any number of '*'. `expand` uses the permuterm index for one star
and the k-gram index otherwise; the tests check that both agree with brute force.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from collections import defaultdict
from typing import Iterable


def _kgrams(s: str, k: int = 2) -> set[str]:
    return {s[i:i + k] for i in range(len(s) - k + 1)}


class WildcardIndex:
    def __init__(self, terms: Iterable[str]):
        self.terms = sorted(set(terms))
        rot = []
        kg: dict[str, set[str]] = defaultdict(set)
        for t in self.terms:
            w = t + "$"
            rot.extend((w[i:] + w[:i], t) for i in range(len(w)))
            for g in _kgrams("$" + t + "$"):
                kg[g].add(t)
        rot.sort()
        self._rot_keys = [r for r, _ in rot]
        self._rot_terms = [t for _, t in rot]
        self._kg = kg

    def permuterm(self, pattern: str) -> list[str]:
        if pattern.count("*") != 1:
            raise ValueError("permuterm handles exactly one '*'")
        x, y = pattern.split("*")
        key = y + "$" + x                                   # X*Y  ->  Y$X*
        i = bisect_left(self._rot_keys, key)
        out = set()
        while i < len(self._rot_keys) and self._rot_keys[i].startswith(key):
            out.add(self._rot_terms[i]); i += 1
        return sorted(out)

    def kgram(self, pattern: str) -> list[str]:
        padded = "$" + pattern + "$"
        grams = set()
        for piece in padded.split("*"):
            grams |= _kgrams(piece)
        if not grams:                                        # pattern is only stars
            return list(self.terms)
        cand = None
        for g in grams:
            s = self._kg.get(g, set())
            cand = set(s) if cand is None else cand & s
            if not cand:
                return []
        rx = re.compile("^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$")
        return sorted(t for t in cand if rx.match(t))

    def expand(self, pattern: str) -> list[str]:
        if "*" not in pattern:
            return [pattern] if pattern in set(self.terms) else []
        return self.permuterm(pattern) if pattern.count("*") == 1 else self.kgram(pattern)
