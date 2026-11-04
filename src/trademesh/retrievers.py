"""Where the app and eval scripts get retrievers and the reranker from.

Contract with the other roles (agree on this, or tell Shashwat the real names):
  * trademesh.index    (Role A)  may define  build_retrievers(passages) -> {name: Retriever}
  * trademesh.ranking  (Role B)  may define  build_retrievers(passages) -> {name: Retriever}
                                       and   build_reranker(passages)   -> Reranker | None
Anything missing is skipped. If nothing at all is found we fall back to the DEV-ONLY BM25 stub
and `using_stub` is True, so the app and the eval scripts can warn or refuse.
"""
from __future__ import annotations

import importlib

from .dev_stub import Bm25Stub
from .types import Passage, Reranker, Retriever


def _try(module: str, attr: str):
    try:
        return getattr(importlib.import_module(f"trademesh.{module}"), attr, None)
    except ImportError:
        return None


def available_retrievers(passages: list[Passage]) -> tuple[dict[str, Retriever], bool]:
    """Return ({name: retriever}, using_stub)."""
    found: dict[str, Retriever] = {}
    for module in ("index", "ranking"):
        build = _try(module, "build_retrievers")
        if build:
            found.update(build(passages))
    if found:
        return found, False
    stub = Bm25Stub(passages)
    return {stub.name: stub}, True


def get_reranker(passages: list[Passage]) -> Reranker | None:
    build = _try("ranking", "build_reranker")
    return build(passages) if build else None
