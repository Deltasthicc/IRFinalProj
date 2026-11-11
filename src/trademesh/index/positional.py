"""Positional inverted index with skip pointers, Boolean, phrase and proximity queries (Role A, proposal 9.1).

    postings: term -> (docs, positions)    docs sorted ascending, positions[i] = sorted token positions in docs[i]

AND is a merge of two sorted lists. With skip pointers every ~sqrt(L) entries the merge can jump over
a run of documents that are all smaller than the other list's current document; `intersect` returns the
number of comparisons so the speed-up is measurable (the lecture's claim) and not just asserted.
A phrase "t1 t2" matches d if some p in pos(t1, d) has p + 1 in pos(t2, d). Stop words are KEPT in this
index, otherwise phrase positions would shift.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from ..ranking.analyzer import Analyzer, AnalyzerConfig
from ..types import Passage
from .wildcard import WildcardIndex

_QTOKEN = re.compile(r'"[^"]*"|\(|\)|[^\s()"]+')


def intersect(a: list[int], b: list[int], skips: bool = True) -> tuple[list[int], int]:
    """Merge-intersect two sorted lists. Returns (result, number of comparisons)."""
    out, i, j, comps = [], 0, 0, 0
    sa = max(1, int(math.sqrt(len(a)))) if skips else 0
    sb = max(1, int(math.sqrt(len(b)))) if skips else 0
    while i < len(a) and j < len(b):
        comps += 1
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            if skips and i % sa == 0 and i + sa < len(a) and a[i + sa] <= b[j]:
                while i % sa == 0 and i + sa < len(a) and a[i + sa] <= b[j]:
                    i += sa
            else:
                i += 1
        else:
            if skips and j % sb == 0 and j + sb < len(b) and b[j + sb] <= a[i]:
                while j % sb == 0 and j + sb < len(b) and b[j + sb] <= a[i]:
                    j += sb
            else:
                j += 1
    return out, comps


class PositionalIndex:
    def __init__(self, passages: list[Passage], analyzer: Analyzer | None = None):
        self.analyzer = analyzer or Analyzer(AnalyzerConfig(stopwords=False))
        self.ids = [p.id for p in passages]
        self.n = len(passages)
        post: dict[str, tuple[list[int], list[list[int]]]] = defaultdict(lambda: ([], []))
        self.lengths = []
        for d, p in enumerate(passages):
            toks = self.analyzer(p.text)
            self.lengths.append(len(toks))
            seen: dict[str, list[int]] = {}
            for pos, t in enumerate(toks):
                seen.setdefault(t, []).append(pos)
            for t, ps in seen.items():
                docs, poss = post[t]
                docs.append(d); poss.append(ps)
        self.postings = dict(post)
        self.wildcards = WildcardIndex(self.postings.keys())
        self.all_docs = list(range(self.n))

    # --- term-level -------------------------------------------------------------------

    def docs(self, term: str) -> list[int]:
        e = self.postings.get(term)
        return e[0] if e else []

    def tf(self, term: str, doc: int) -> int:
        e = self.postings.get(term)
        if not e:
            return 0
        from bisect import bisect_left
        i = bisect_left(e[0], doc)
        return len(e[1][i]) if i < len(e[0]) and e[0][i] == doc else 0

    def expand_wildcard(self, pattern: str) -> list[str]:
        return self.wildcards.expand(pattern)

    # --- phrase and proximity ------------------------------------------------------------

    def phrase(self, terms: list[str]) -> list[int]:
        """Documents containing the terms as consecutive tokens."""
        return self.near(terms, 1)

    def near(self, terms: list[str], gap: int = 1) -> list[int]:
        """Documents where each next term occurs within `gap` tokens AFTER the previous one (gap = 1: a phrase)."""
        if not terms or any(t not in self.postings for t in terms):
            return []
        cands = self.docs(terms[0])
        for t in terms[1:]:
            cands, _ = intersect(cands, self.docs(t))
        out = []
        for d in cands:
            frontier = set(self._pos(terms[0], d))
            for t in terms[1:]:
                nxt = set(self._pos(t, d))
                frontier = {q for q in nxt if any(0 < q - p <= gap for p in frontier)}
                if not frontier:
                    break
            if frontier:
                out.append(d)
        return out

    def _pos(self, term: str, doc: int) -> list[int]:
        from bisect import bisect_left
        docs, poss = self.postings[term]
        return poss[bisect_left(docs, doc)]

    # --- Boolean queries ---------------------------------------------------------------------

    def boolean(self, query: str) -> list[int]:
        """AND, OR, NOT (case-sensitive upper case), parentheses, "quoted phrases", wildcard* terms.
        Precedence NOT > AND > OR; adjacent terms mean AND."""
        toks = _QTOKEN.findall(query)
        pos = 0

        def peek():
            return toks[pos] if pos < len(toks) else None

        def parse_or():
            nonlocal pos
            res = parse_and()
            while peek() == "OR":
                pos += 1
                res = _union(res, parse_and())
            return res

        def parse_and():
            nonlocal pos
            res = parse_not()
            while peek() is not None and peek() not in ("OR", ")"):
                if peek() == "AND":
                    pos += 1
                res = intersect(res, parse_not())[0]
            return res

        def parse_not():
            nonlocal pos
            if peek() == "NOT":
                pos += 1
                return _minus(self.all_docs, parse_not())
            return parse_atom()

        def parse_atom():
            nonlocal pos
            t = peek()
            if t is None:
                raise ValueError("unexpected end of query")
            pos += 1
            if t == "(":
                res = parse_or()
                if peek() != ")":
                    raise ValueError("missing )")
                pos += 1
                return res
            if t.startswith('"'):
                return self.phrase(self.analyzer(t.strip('"')))
            if "*" in t:
                res: list[int] = []
                for term in self.expand_wildcard(t.lower()):
                    res = _union(res, self.docs(term))
                return res
            parts = self.analyzer(t)
            res = self.docs(parts[0]) if parts else []
            for extra in parts[1:]:                       # e.g. "383,285" or "co-op" tokenised into several pieces
                res = intersect(res, self.docs(extra))[0]
            return res

        if not toks:
            return []
        result = parse_or()
        if pos != len(toks):
            raise ValueError(f"unexpected {toks[pos]!r}")
        return result


def _union(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) | set(b))


def _minus(a: list[int], b: list[int]) -> list[int]:
    drop = set(b)
    return [x for x in a if x not in drop]
