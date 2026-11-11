"""Role B: ranking. Auto-discovered by trademesh.retrievers (see that module for the contract).

    build_retrievers(passages) -> {name: Retriever}  bm25, bm25+prf, dense, hybrid (when ML deps exist)
    build_reranker(passages)   -> Reranker | None    cross-encoder, or None when unavailable

Dense and cross-encoder systems need `sentence-transformers` (requirements-ml.txt). Without it we
return the lexical systems only, and say so on stderr rather than pretending.
"""
from __future__ import annotations

import sys

from .analyzer import Analyzer, AnalyzerConfig
from .bm25 import BM25Index
from .dense import DenseRetriever, HashingEncoder, SentenceTransformerEncoder
from .fusion import HybridRetriever, combsum, rrf
from .prf import PRFRetriever
from .rerank import CrossEncoderReranker, CrossEncoderScorer

__all__ = ["Analyzer", "AnalyzerConfig", "BM25Index", "DenseRetriever", "HashingEncoder",
           "SentenceTransformerEncoder", "HybridRetriever", "PRFRetriever", "CrossEncoderReranker",
           "CrossEncoderScorer", "rrf", "combsum", "build_retrievers", "build_reranker"]

# Defaults fixed in the proposal; k1/b are re-tuned on the dev split only (eval/tune_bm25.py)
# and written to data/judgments/bm25_params.json, which is read here when present.
DEFAULT_CONFIG = AnalyzerConfig(stopwords=True, stem=False, keep_numbers=True, add_metadata=False)


def _params() -> tuple[float, float]:
    import json
    import os
    from pathlib import Path
    f = Path(os.environ.get("TRADEMESH_JUDGMENTS", "data/judgments")) / "bm25_params.json"
    if f.exists():
        d = json.loads(f.read_text())
        return float(d["k1"]), float(d["b"])
    return 1.2, 0.75


def build_retrievers(passages):
    k1, b = _params()
    bm25 = BM25Index(passages, Analyzer(DEFAULT_CONFIG), k1=k1, b=b, name="bm25")
    systems = {bm25.name: bm25}
    prf = PRFRetriever(bm25)
    systems[prf.name] = prf
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("[ranking] sentence-transformers not installed: dense, hybrid and rerank are skipped "
              "(pip install -r requirements-ml.txt)", file=sys.stderr)
        return systems
    dense = DenseRetriever(passages, SentenceTransformerEncoder(), name="dense")
    hybrid = HybridRetriever([bm25, dense], name="hybrid")
    systems.update({dense.name: dense, hybrid.name: hybrid})
    return systems


def build_reranker(passages):
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    return CrossEncoderReranker(passages)
