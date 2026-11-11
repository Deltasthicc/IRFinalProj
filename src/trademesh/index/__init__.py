"""Role A: the classic IR core. Auto-discovered by trademesh.retrievers.

    build_retrievers(passages) -> {name: Retriever}
        tfidf              ltc cosine, heap top-K (System 1)
        tfidf-champ-50     the same with champion lists of 50 per term
        tfidf-champ-200    ... of 200 per term
        boolean            unranked Boolean/phrase engine, matches ordered by term count
"""
from __future__ import annotations

from .boolean import BooleanRetriever
from .positional import PositionalIndex, intersect
from .tfidf import TfidfIndex
from .wildcard import WildcardIndex

__all__ = ["PositionalIndex", "TfidfIndex", "BooleanRetriever", "WildcardIndex", "intersect", "build_retrievers"]

CHAMPION_SIZES = (50, 200)


def build_retrievers(passages):
    systems = {"tfidf": TfidfIndex(passages)}
    for r in CHAMPION_SIZES:
        c = TfidfIndex(passages, champions=r)
        systems[c.name] = c
    systems["boolean"] = BooleanRetriever(PositionalIndex(passages))
    return systems
