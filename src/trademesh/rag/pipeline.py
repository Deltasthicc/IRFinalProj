"""End-to-end query -> retrieve -> context -> LLM -> verify -> abstain pipeline.

    question
      -> entity detection + spelling correction            (query.entities)
      -> topic query = question without company mentions
      -> retrieve top candidates, keep the detected company (entity filter)
      -> optional section prior, optional reranker
      -> top-5 context [P1]..[P5] -> LLM (cached, temperature 0)
      -> verifier -> abstention decision

Compare mode: when two or more companies are detected, one sub-query per company is run
and the results are merged (grouped by company) into one context.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..query.entities import EntityMatcher, Mention
from ..query.section_prior import apply_prior
from ..types import Hit, Passage, Reranker, Retriever
from ..text import content_terms
from .abstain import confidence, decide
from .llm import LLMClient
from .prompt import (CLOSED_BOOK_SYSTEM, RAG_SYSTEM, build_closed_book_prompt, build_user_prompt, is_insufficient)
from .verifier import VerifierResult, verify

FILTER_DEPTH = 300      # candidates fetched when we need to filter by company


@dataclass
class RagConfig:
    k_context: int = 5          # passages given to the LLM (single-company mode)
    k_per_company: int = 3      # passages per company in compare mode
    k_retrieve: int = 30        # candidates passed to the reranker
    lam: float = 0.0            # section prior weight (tuned on dev; 0 = off)
    tau: float = 0.0            # abstention threshold on `confidence` (tuned on dev; 0 = off)
    entity_filter: bool = True
    call_llm_when_low_score: bool = False   # eval scripts set True to log the model's own answer too


@dataclass
class RagResult:
    question: str
    mode: str                                   # "rag" | "compare" | "closed_book"
    answer: str = ""
    abstained: bool = False
    abstain_reason: str = ""
    passages: list[Passage] = field(default_factory=list)       # context, in [P1].. order
    hits: list[Hit] = field(default_factory=list)               # scores of those passages
    by_company: dict[str, list[int]] = field(default_factory=dict)   # ticker -> 1-based labels (compare)
    mentions: list[Mention] = field(default_factory=list)
    corrections: list[tuple[str, str]] = field(default_factory=list)
    topic_query: str = ""
    confidence: float = 0.0
    model_said_insufficient: bool = False
    verifier: VerifierResult | None = None
    cached: bool | None = None
    model: str = ""
    provider: str = ""


class RagPipeline:
    def __init__(self, passages: list[Passage], retriever: Retriever, llm: LLMClient,
                 matcher: EntityMatcher, reranker: Reranker | None = None, config: RagConfig | None = None):
        self.by_id = {p.id: p for p in passages}
        self.retriever, self.llm, self.matcher, self.reranker = retriever, llm, matcher, reranker
        self.cfg = config or RagConfig()

    # --- retrieval -------------------------------------------------------------------

    def retrieve(self, query: str, ticker: str | None = None, k: int | None = None) -> list[Hit]:
        return self._retrieve(query, ticker, k)[0]

    def _retrieve(self, query: str, ticker: str | None = None, k: int | None = None) -> tuple[list[Hit], list[Hit]]:
        """(final hits, base hits). Candidates for `query`, optionally restricted to one
        company, then section prior, then reranker. The base hits (before prior and rerank)
        are what the abstention confidence is computed from."""
        cfg = self.cfg
        k = k or cfg.k_context
        hits = self.retriever.search(query, k=FILTER_DEPTH if ticker else cfg.k_retrieve)
        if ticker:
            hits = [h for h in hits if self.by_id[h.passage_id].ticker == ticker][: cfg.k_retrieve]
        base = hits
        hits = apply_prior(hits, self.by_id, cfg.lam)
        if self.reranker is not None and hits:
            hits = self.reranker.rerank(query, hits, k=k)
        return hits[:k], base

    # --- main entry ------------------------------------------------------------------

    def answer(self, question: str, closed_book: bool = False, compare: bool | None = None) -> RagResult:
        if closed_book:
            return self._closed_book(question)
        mentions = self.matcher.find_mentions(question)
        tickers = list(dict.fromkeys(m.ticker for m in mentions)) if self.cfg.entity_filter else []
        topic = self.matcher.strip_mentions(question, mentions) if tickers else question
        if not content_terms(topic):
            topic = question
        res = RagResult(question, "rag", mentions=mentions, topic_query=topic,
                        corrections=self.matcher.corrections(mentions))

        is_compare = (len(tickers) >= 2) if compare is None else compare
        if is_compare and len(tickers) >= 2:
            res.mode = "compare"
            hits, confs = [], []
            for t in tickers:
                sub, base = self._retrieve(topic, ticker=t, k=self.cfg.k_per_company)
                res.by_company[t] = list(range(len(hits) + 1, len(hits) + len(sub) + 1))
                hits += sub
                confs.append(confidence(base, topic))
            res.confidence = min(confs)      # the weakest company limits how far we can trust the answer
        else:
            ticker = tickers[0] if len(tickers) == 1 else None
            hits, base = self._retrieve(topic, ticker=ticker)
            if not hits and ticker:             # company not in the index: fall back to no filter
                hits, base = self._retrieve(topic)
            res.confidence = confidence(base, topic)

        res.hits = hits
        res.passages = [self.by_id[h.passage_id] for h in hits]

        low = res.confidence < self.cfg.tau
        if not res.passages or (low and not self.cfg.call_llm_when_low_score):
            d = decide(None, res.confidence, self.cfg.tau) if res.passages else None
            res.abstained = True
            res.abstain_reason = d.reason if d else "no passage was retrieved"
            res.answer = "INSUFFICIENT EVIDENCE"
            return res

        r = self.llm.complete(RAG_SYSTEM, build_user_prompt(question, res.passages))
        res.answer, res.cached, res.model, res.provider = r.text, r.cached, r.model, r.provider
        res.model_said_insufficient = is_insufficient(r.text)
        res.verifier = verify(r.text, res.passages)
        d = decide(r.text, res.confidence, self.cfg.tau)
        res.abstained, res.abstain_reason = d.abstain, d.reason
        return res

    def _closed_book(self, question: str) -> RagResult:
        r = self.llm.complete(CLOSED_BOOK_SYSTEM, build_closed_book_prompt(question))
        return RagResult(question, "closed_book", answer=r.text, cached=r.cached, model=r.model, provider=r.provider,
                         topic_query=question)
