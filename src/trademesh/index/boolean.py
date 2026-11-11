"""The Boolean retriever used as a comparison system in RQ1 (Role A).

Plain keyword queries mean AND over the content words (stop words dropped), which is what an unranked
Boolean engine does; queries using AND/OR/NOT, quotes, parentheses or '*' are parsed as written. A
Boolean engine returns a SET, not a ranking. To fit the Retriever interface we order the matches by the
total count of the query terms in the passage (documented tie-break), and give each match that count as
its score. A query whose terms never co-occur returns nothing: that is the behaviour being measured.
"""
from __future__ import annotations

import re

from ..ranking.analyzer import STOP_WORDS
from ..types import Hit
from .positional import PositionalIndex

_OPS = re.compile(r'\b(AND|OR|NOT)\b|["()*]')


class BooleanRetriever:
    name = "boolean"

    def __init__(self, index: PositionalIndex):
        self.index = index

    def search(self, query: str, k: int = 30) -> list[Hit]:
        ix = self.index
        if _OPS.search(query):
            docs = ix.boolean(query)
            terms = [t for t in ix.analyzer(re.sub(r'\b(AND|OR|NOT)\b|[()"*]', " ", query))]
        else:
            terms = [t for t in ix.analyzer(query) if t not in STOP_WORDS]
            docs = ix.boolean(" AND ".join(terms)) if terms else []
        scored = [(sum(ix.tf(t, d) for t in set(terms)) or 1, d) for d in docs]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [Hit(ix.ids[d], float(s)) for s, d in scored[:k]]
