"""App back end without any Streamlit code, so it can be tested offline.

Optional data from Role A (the app shows "not available yet" when a file is missing or malformed):

  data/graph/company_graph.json
      {"nodes": [{"ticker": "AAPL", "pagerank": 0.07, "hub": 0.1, "authority": 0.2, "in_degree": 5}, ...],
       "edges": [{"source": "AAPL", "target": "TSLA", "weight": 3}, ...]}      # source mentions target

  data/vision/results.json
      {"queries": [{"query": "...", "results": [{"ticker": "AAPL", "page": 41, "score": 0.83,
                                                 "image": "data/vision/pages/AAPL_41.png"}, ...]}]}

These formats are C's assumption; if A writes something different, change load_graph/load_vision only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..data import DATA, ROOT, load_corpus
from ..query.entities import EntityMatcher
from ..rag.llm import LLMClient
from ..rag.pipeline import RagConfig, RagPipeline
from ..retrievers import available_retrievers, get_reranker
from ..text import tokenize
from ..types import Passage, Reranker, Retriever

GRAPH_PATH = DATA / "graph" / "company_graph.json"
VISION_PATH = DATA / "vision" / "results.json"


@dataclass
class AppState:
    passages: list[Passage]
    companies: list[dict]
    is_fixture: bool
    retrievers: dict[str, Retriever]
    using_stub: bool
    reranker: Reranker | None
    matcher: EntityMatcher
    llm: LLMClient

    def pipeline(self, retriever: str, cfg: RagConfig, use_reranker: bool = True) -> RagPipeline:
        return RagPipeline(self.passages, self.retrievers[retriever], self.llm, self.matcher,
                           self.reranker if use_reranker else None, cfg)

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.is_fixture:
            out.append("Running on the built-in TEST FIXTURE (invented text), not the real filings. "
                       "Do not read anything here as a result.")
        if self.using_stub:
            out.append("Using the development BM25 stub, since the real retrievers are not installed yet.")
        if self.llm.is_stub:
            out.append("LLM_PROVIDER is 'stub': answers are extractive quotes, not a language model. "
                       "Set a provider in .env.")
        return out


def load_state() -> AppState:
    passages, companies, is_fixture = load_corpus()
    retrievers, using_stub = available_retrievers(passages)
    vocab = {t for p in passages for t in tokenize(p.text)}
    return AppState(passages, companies, is_fixture, retrievers, using_stub, get_reranker(passages),
                    EntityMatcher(companies, known_words=vocab), LLMClient())


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path, validate) -> tuple[dict | None, str]:
    if not path.exists():
        return None, f"not available yet: {_rel(path)} has not been produced."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        validate(data)
        return data, ""
    except Exception as e:  # malformed file must never crash the app
        return None, f"could not read {_rel(path)}: {e}"


def _validate_graph(d: dict) -> None:
    assert isinstance(d["nodes"], list) and isinstance(d["edges"], list)
    for n in d["nodes"]:
        n["ticker"], n["pagerank"]


def _validate_vision(d: dict) -> None:
    assert isinstance(d["queries"], list)


RESULT_DIRS = [DATA / "results", ROOT / "eval" / "out"]      # never eval/out/dryrun: dry runs are not results


def load_results() -> tuple[dict | None, str]:
    """Locked retrieval results for the Evaluation tab: retrieval_test.json (+ hypotheses_test.json) from
    data/results/ or eval/out/. Falls back to the dev run, labelled as such."""
    for split in ("test", "dev"):
        for d in RESULT_DIRS:
            f = d / f"retrieval_{split}.json"
            if f.exists():
                data, why = _load_json(f, lambda x: x["overall"])
                if data is None:
                    return None, why
                h = d / f"hypotheses_{split}.json"
                data["split"] = split
                data["hypotheses"] = json.loads(h.read_text()) if h.exists() else []
                return data, ""
    return None, "not available yet: run eval/run_retrieval.py after judging (see docs/report_B/README.md)."


def load_rag_study() -> tuple[dict | None, str]:
    """RAG answer ratings, paired bootstrap and abstention results for the Evaluation tab (test split).
    Needs eval/out/rag_scores_test.json; the bootstrap and abstention files are optional."""
    def find(name: str) -> Path | None:
        return next((d / name for d in RESULT_DIRS if (d / name).exists()), None)

    scores = find("rag_scores_test.json")
    if scores is None:
        return None, "not available yet: run eval/rag_score.py (see docs/report_C/audit/README.md)."
    out: dict = {}
    for key, name in (("scores", "rag_scores_test.json"), ("bootstrap", "rag_bootstrap_test.json"),
                      ("abstention", "abstention_eval.json"), ("abstention_score_only", "abstention_scoreonly.json")):
        f = find(name)
        if f is None:
            out[key] = None
            continue
        try:
            out[key] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # a malformed file must never crash the app
            return None, f"could not read {_rel(f)}: {e}"
    if not out["scores"].get("configs"):
        return None, f"could not read {_rel(scores)}: no 'configs' entry."
    return out, ""


def load_graph(path: Path = GRAPH_PATH) -> tuple[dict | None, str]:
    return _load_json(path, _validate_graph)


def load_vision(path: Path = VISION_PATH) -> tuple[dict | None, str]:
    return _load_json(path, _validate_vision)


def graph_dot(graph: dict, top: int = 20) -> str:
    """Graphviz DOT for the top-`top` companies by PageRank (node size ~ PageRank)."""
    nodes = sorted(graph["nodes"], key=lambda n: -n["pagerank"])[:top]
    keep = {n["ticker"] for n in nodes}
    top_pr = max((n["pagerank"] for n in nodes), default=1) or 1
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=circle, style=filled, fillcolor="#d7ecff", fontsize=10];']
    for n in nodes:
        w = 0.4 + 0.8 * n["pagerank"] / top_pr
        lines.append(f'"{n["ticker"]}" [width={w:.2f}, fixedsize=true];')
    for e in graph["edges"]:
        if e["source"] in keep and e["target"] in keep:
            lines.append(f'"{e["source"]}" -> "{e["target"]}" [penwidth={min(4, 0.5 + e.get("weight", 1) * 0.3):.1f}];')
    lines.append("}")
    return "\n".join(lines)


def highlight_citations(answer: str) -> str:
    """Bold the [Pn] markers so they stand out in the markdown answer."""
    import re
    return re.sub(r"\[P(\d+)\]", r"**[P\1]**", answer)
