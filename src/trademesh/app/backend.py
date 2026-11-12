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


# The same questions eval/demo_warm.py puts in the answer cache, so they work offline once it is warm.
EXAMPLES = [
    ("Typo / CEO risk", "How does Telsa describe the risk of depending on its Chief Executive Officer?"),
    ("Compare / cloud", "Compare how Microsoft and Amazon describe competition in cloud services."),
    ("Figure / segments", "What was Microsoft's revenue by segment in the latest fiscal year?"),
    ("Trap / future year", "What was Apple's net sales in fiscal year 2031?"),
]

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
    """Graphviz DOT for the top-`top` companies by PageRank (node size ~ PageRank), in the app's dark palette."""
    nodes = sorted(graph["nodes"], key=lambda n: -n["pagerank"])[:top]
    keep = {n["ticker"] for n in nodes}
    top_pr = max((n["pagerank"] for n in nodes), default=1) or 1
    lines = ["digraph G {", "rankdir=LR;", 'bgcolor="transparent";',
             'node [shape=circle, style=filled, fillcolor="#101218", color="#EDEBE6", fontcolor="#EDEBE6", fontsize=10];',
             'edge [color="#6B6E7B"];']
    for n in nodes:
        w = 0.4 + 0.8 * n["pagerank"] / top_pr
        lines.append(f'"{n["ticker"]}" [width={w:.2f}, fixedsize=true];')
    for e in graph["edges"]:
        if e["source"] in keep and e["target"] in keep:
            lines.append(f'"{e["source"]}" -> "{e["target"]}" [penwidth={min(4, 0.5 + e.get("weight", 1) * 0.3):.1f}];')
    lines.append("}")
    return "\n".join(lines)


def md_safe(text: str) -> str:
    """Escape dollar signs so Streamlit markdown does not treat money amounts as LaTeX."""
    return text.replace("\\$", "$").replace("$", "\\$")


def highlight_citations(answer: str) -> str:
    """Bold the [Pn] markers so they stand out in the markdown answer (and escape dollar signs)."""
    import re
    return re.sub(r"\[P(\d+)\]", r"**[P\1]**", md_safe(answer))


ENTITY_EVAL = ROOT / "eval" / "out" / "entity_eval.json"


def load_entity_eval(path: Path = ENTITY_EVAL) -> dict | None:
    """Result of eval/entity_eval.py, if it has been run (real numbers only; never invented)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def readiness(state: AppState) -> list[tuple[str, bool, str]]:
    """What is wired up and what is still missing (name, ready, detail)."""
    graph_ok = load_graph()[0] is not None
    vision_ok = load_vision()[0] is not None
    return [
        ("Corpus", not state.is_fixture, f"{len(state.passages):,} passages" if not state.is_fixture else "using the test fixture"),
        ("Retrievers", not state.using_stub, ", ".join(state.retrievers) if not state.using_stub else "dev BM25 stub"),
        ("Reranker", state.reranker is not None, "cross-encoder installed" if state.reranker else "not installed (pip install -r requirements-ml.txt)"),
        ("LLM", not state.llm.is_stub, f"{state.llm.provider} / {state.llm.model}"),
        ("Company graph", graph_ok, "data/graph/company_graph.json" if graph_ok else "not produced yet"),
        ("Page images", vision_ok, "data/vision/results.json" if vision_ok else "vision track is gated, not run"),
    ]


def headline_numbers() -> list[tuple[str, str, str, bool]]:
    """Four bento tiles for the System tab, built only from result files that exist.
    Each is (label, big value, sub text, pending). Missing results show as pending, never as a guess."""
    rag, _ = load_rag_study()
    ret, _ = load_results()
    ent = load_entity_eval()
    tiles: list[tuple[str, str, str, bool]] = []
    cfg = (rag or {}).get("scores", {}).get("configs", {})
    if "best_rag" in cfg and "closed_book" in cfg:
        tiles.append(("Answer quality, best-RAG (0 to 2)", f'{cfg["best_rag"]["mean_rating"]:.2f}',
                      f'vs {cfg["closed_book"]["mean_rating"]:.2f} closed-book, 20 test questions, LLM-rated', False))
    else:
        tiles.append(("Answer quality, best-RAG", "-", "RAG study not run yet", True))
    tiles.append(("Entity accuracy", f'{ent["accuracy"] * 100:.1f}%' if ent else "-",
                  f'{ent["correct"]}/{ent["n"]} misspelt names' if ent else "run eval/entity_eval.py", ent is None))
    ab = (rag or {}).get("abstention")
    if ab:
        t = ab["test"]
        tiles.append(("Unanswerable declined", f'{t["tp"]}/{t["tp"] + t["fn"]}', f'{t["fp"]} wrongly declined of {t["fp"] + t["tn"]} answerable', False))
    else:
        tiles.append(("Unanswerable declined", "-", "abstention study not run yet", True))
    if ret and ret.get("overall"):
        best = max(ret["overall"], key=lambda k: ret["overall"][k].get("nDCG@10", 0))
        base = ret.get("baseline")
        sub = f'nDCG@10, {best}' + (f' vs {ret["overall"][base]["nDCG@10"]:.3f} {base}' if base in ret["overall"] and base != best else "")
        tiles.append(("Best retrieval", f'{ret["overall"][best]["nDCG@10"]:.3f}', f'{sub}, {ret["n_needs"]} {ret["split"]} needs', False))
    else:
        tiles.append(("Best retrieval", "-", "retrieval study not run yet", True))
    return tiles
