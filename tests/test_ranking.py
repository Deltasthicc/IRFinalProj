"""Role B tests. Expected values are worked out by hand (comments show the arithmetic)."""
import math

import numpy as np
import pytest

from trademesh.evaluation import metrics as m
from trademesh.evaluation.pooling import assign_judges, build_pool
from trademesh.evaluation.qrels import consolidate, weighted_kappa
from trademesh.evaluation.stats import paired_bootstrap
from trademesh.ranking import (Analyzer, AnalyzerConfig, BM25Index, CrossEncoderReranker, DenseRetriever,
                               HashingEncoder, HybridRetriever, PRFRetriever, build_retrievers, rrf)
from trademesh.ranking.fusion import combsum
from trademesh.types import Hit, Passage


def P(i, text, ticker="XYZ", company="Xyz Corp"):
    return Passage(id=i, ticker=ticker, company=company, fiscal_year=2025, item="1A", kind="text", text=text)


# ---------------------------------------------------------------- analyzer

def test_analyzer_numbers_stopwords_stemming():
    a = Analyzer(AnalyzerConfig(stopwords=True, stem=False, keep_numbers=True))
    assert a("Net sales were $383,285 million, up 5.2% in the year") == ["net", "sales", "383285", "million", "5.2", "year"]
    assert "383285" not in Analyzer(AnalyzerConfig(keep_numbers=False))("sales of 383,285")
    assert Analyzer(AnalyzerConfig(stopwords=False))("the risk")[0] == "the"
    assert Analyzer(AnalyzerConfig(stem=True))("supply risks disrupting") == ["suppli", "risk", "disrupt"]


def test_analyzer_metadata_prefix():
    on = Analyzer(AnalyzerConfig(add_metadata=True))
    assert on.passage_text("AAPL", "Apple Inc.", "x") == "AAPL Apple Inc. x"
    assert Analyzer().passage_text("AAPL", "Apple Inc.", "x") == "x"


# ---------------------------------------------------------------- BM25

DOCS = [P("d1", "tariffs raise costs"), P("d2", "tariffs tariffs tariffs and a long list of other unrelated words today"),
        P("d3", "supply chain disruption")]


def test_bm25_matches_hand_computation():
    ix = BM25Index(DOCS, Analyzer(AnalyzerConfig(stopwords=False)), k1=1.2, b=0.75)
    # N=3, df(tariffs)=2 -> idf = ln((3-2+0.5)/(2+0.5)+1) = ln(1.6)
    idf = math.log(1.6)
    assert ix.idf("tariffs") == pytest.approx(idf)
    # |d1|=3, |d2|=12, |d3|=3 -> avgdl = 6; d1: tf=1, norm = 0.25+0.75*3/6 = 0.625
    expected = idf * 1 * 2.2 / (1 + 1.2 * 0.625)
    hits = ix.search("tariffs")
    scores = {h.passage_id: h.score for h in hits}
    assert scores["d1"] == pytest.approx(expected)
    # d2: tf=3, norm = 0.25 + 0.75*12/6 = 1.75 -> BM25 saturates tf, but 3 occurrences still beat 1
    d2 = idf * 3 * 2.2 / (3 + 1.2 * 1.75)
    assert scores["d2"] == pytest.approx(d2) and d2 < 3 * expected     # saturation: far less than 3x
    assert [h.passage_id for h in hits] == ["d2", "d1"]
    assert ix.search("nonexistentterm") == []


def test_bm25_explain_sums_to_score_and_params_update():
    ix = BM25Index(DOCS, Analyzer(AnalyzerConfig(stopwords=False)))
    score = ix.search("tariffs costs", k=5)[0].score
    assert sum(w for _, w in ix.explain("tariffs costs", "d1")) == pytest.approx(score)
    ix.set_params(2.0, 0.3)
    assert ix.search("tariffs", 1)[0].score != pytest.approx(score)


def test_bm25_is_deterministic_on_ties():
    docs = [P(f"t{i}", "same words") for i in range(5)]
    a = [h.passage_id for h in BM25Index(docs).search("same words", 3)]
    assert a == ["t0", "t1", "t2"]


# ---------------------------------------------------------------- fusion

def test_rrf_hand_values():
    r1, r2 = [Hit("a", 9), Hit("b", 8), Hit("c", 7)], [Hit("c", 3), Hit("a", 2)]
    out = rrf([r1, r2], k=60)
    # a: 1/61 + 1/62 = 0.032522 ; c: 1/63 + 1/61 = 0.032266 ; b: 1/62 = 0.016129
    assert [h.passage_id for h in out] == ["a", "c", "b"]
    assert out[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_combsum_and_mnz():
    r1, r2 = [Hit("a", 10), Hit("b", 0)], [Hit("a", 1), Hit("c", 0)]
    s = {h.passage_id: h.score for h in combsum([r1, r2])}
    assert s == {"a": 2.0, "b": 0.0, "c": 0.0}
    assert {h.passage_id: h.score for h in combsum([r1, r2], mnz=True)}["a"] == 4.0


def test_hybrid_finds_what_either_system_finds():
    docs = [P("d1", "tariffs raise costs"), P("d2", "supply chain disruption risk"), P("d3", "dividend policy")]
    bm = BM25Index(docs)
    de = DenseRetriever(docs, HashingEncoder(), cache_dir=None)
    hy = HybridRetriever([bm, de])
    assert hy.search("tariffs costs", 3)[0].passage_id == "d1"
    with pytest.raises(ValueError):
        HybridRetriever([bm], method="nope")


# ---------------------------------------------------------------- dense + rerank

def test_dense_cosine_and_cache(tmp_path):
    docs = [P("d1", "tariffs raise costs"), P("d2", "dividend policy")]
    d = DenseRetriever(docs, HashingEncoder(), cache_dir=tmp_path)
    assert d.search("tariffs raise costs", 2)[0].passage_id == "d1"
    assert d.search("tariffs raise costs", 1)[0].score == pytest.approx(1.0, abs=1e-5)
    assert len(list(tmp_path.glob("emb_*.npy"))) == 1
    d2 = DenseRetriever(docs, HashingEncoder(), cache_dir=tmp_path)           # loaded from cache
    assert np.allclose(d.matrix, d2.matrix)


def test_reranker_reorders_and_calibrates():
    docs = [P("d1", "aaa"), P("d2", "bbb"), P("d3", "ccc")]
    scorer = lambda pairs: [{"aaa": -2.0, "bbb": 3.0, "ccc": 0.0}[t] for _, t in pairs]
    rr = CrossEncoderReranker(docs, scorer=scorer, depth=3)
    out = rr.rerank("q", [Hit("d1", 1), Hit("d2", 1), Hit("d3", 1)], k=2)
    assert [h.passage_id for h in out] == ["d2", "d3"]
    assert out[0].score == pytest.approx(1 / (1 + math.exp(-3)))
    assert rr.rerank("q", [], k=2) == []


# ---------------------------------------------------------------- PRF

def test_prf_adds_terms_from_top_documents():
    docs = [P("d1", "tariffs duties import costs margin"), P("d2", "tariffs duties import costs squeeze"),
            P("d3", "duties import costs rising"), P("d4", "dividend policy board")]
    ix = BM25Index(docs)
    prf = PRFRetriever(ix, k_docs=2, m_terms=3)
    weights, added = prf.expand("tariffs")
    assert "tariffs" in weights and weights["tariffs"] >= max(w for t, w in weights.items() if t != "tariffs")
    assert set(added) <= {"duties", "import", "costs", "margin", "squeeze"} and added
    # the expansion lets d3 (no 'tariffs') be retrieved
    assert "d3" in [h.passage_id for h in prf.search("tariffs", 5)]
    assert "d3" not in [h.passage_id for h in ix.search("tariffs", 5)]


# ---------------------------------------------------------------- discovery contract

def test_build_retrievers_returns_named_dict(passages):
    systems = build_retrievers(passages)
    assert isinstance(systems, dict) and {"bm25", "bm25+prf"} <= set(systems)
    assert systems["bm25"].search("tariffs", 3)


# ---------------------------------------------------------------- metrics (hand-checked)

def test_ndcg_matches_lecture_numbers():
    # DCG = 3/1 + 2/log2(3) + 1/log2(4) + 0 + 1/log2(6) = 5.1488 ; ideal [3,2,1,1,0] = 5.1926 -> 0.9916
    rels = {"a": 3, "b": 2, "c": 1, "e": 1}
    ranked = ["a", "b", "c", "x", "e"]
    assert m.dcg([3, 2, 1, 0, 1]) == pytest.approx(5.1488, abs=1e-3)
    assert m.ndcg_at_k(ranked, rels, 5) == pytest.approx(5.1488 / 5.1926, abs=1e-3)
    # binary gain: grades 3,2,1,1 all count 1 -> DCG 1+1/1.585+1/2+1/2.585 (rank 4 empty), ideal 4 relevant
    assert m.ndcg_at_k(["a", "x", "b"], {"a": 2, "b": 1, "c": 1}, 3, binary=True) == pytest.approx(
        (1 + 0 + 1 / 2) / (1 + 1 / math.log2(3) + 1 / 2))
    # lecture example: grades [3,2,3,0,1,2] -> DCG 6.861
    assert m.dcg([3, 2, 3, 0, 1, 2]) == pytest.approx(6.861, abs=1e-3)


def test_binary_metrics():
    rels = {"a": 2, "c": 1, "e": 1, "z": 0}
    ranked = ["a", "b", "c", "d"]
    assert m.precision_at_k(ranked, rels, 4) == 0.5
    assert m.recall_at_k(ranked, rels, 4) == pytest.approx(2 / 3)
    assert m.f1_at_k(ranked, rels, 4) == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert m.average_precision(ranked, rels) == pytest.approx((1 + 2 / 3) / 3)    # e never retrieved
    assert m.reciprocal_rank(["b", "c"], rels) == 0.5
    assert m.success_at_k(["b", "d"], rels, 2) == 0.0
    assert m.average_precision(["a"], {}) == 0.0


def test_bpref_ignores_unjudged():
    rels = {"a": 1, "c": 1, "b": 0, "d": 0}                     # R = 2, N = 2
    # ranked b a d c: a has 1 judged nonrel above -> 1-1/2 ; c has 2 -> 1-2/2 ; bpref = (0.5+0)/2
    assert m.bpref(["b", "a", "d", "c"], rels) == pytest.approx(0.25)
    # an unjudged document in front changes nothing
    assert m.bpref(["u", "b", "a", "d", "c"], rels) == pytest.approx(0.25)
    assert m.judged_at_k(["u", "a", "b", "q"], rels, 4) == 0.5


def test_interpolated_11pt():
    rels = {"a": 1, "c": 1}
    # x a y c: precision 1/2 at recall .5 and 2/4 at recall 1 -> 0.5 at every level
    assert m.interpolated_11pt(["x", "a", "y", "c"], rels) == [0.5] * 11
    assert m.interpolated_11pt(["a", "c"], rels) == [1.0] * 11
    assert m.interpolated_11pt(["x"], rels) == [0.0] * 11


def test_rank_correlation():
    assert m.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert m.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert m.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)               # ties get average ranks
    assert m.kendall_tau([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)          # 2 concordant, 1 discordant


# ---------------------------------------------------------------- stats, pooling, qrels

def test_bootstrap_identical_and_clearly_better():
    same = paired_bootstrap([0.5] * 10, [0.5] * 10)
    assert same["mean_diff"] == 0 and same["ties"] == 10 and same["p"] > 0.9
    better = paired_bootstrap([0.9, 0.8, 0.7, 0.9, 0.8, 0.9], [0.1, 0.2, 0.1, 0.3, 0.2, 0.1])
    assert better["ci_low"] > 0 and better["wins"] == 6 and better["p"] < 0.01
    assert paired_bootstrap([0.2] * 5, [0.1] * 5, seed=3) == paired_bootstrap([0.2] * 5, [0.1] * 5, seed=3)


def test_pool_is_deduplicated_shuffled_and_reproducible():
    runs = {"s1": [Hit("a", 1), Hit("b", 1), Hit("c", 1)], "s2": [Hit("b", 1), Hit("d", 1), Hit("a", 1)]}
    ids, contrib = build_pool(runs, depth=2, seed=1)
    assert sorted(ids) == ["a", "b", "d"] and contrib["b"] == ["s1", "s2"] and contrib["d"] == ["s2"]
    assert build_pool(runs, depth=2, seed=1)[0] == ids
    judges = assign_judges([f"q{i}" for i in range(20)], ["A", "B", "C"], n_double=10)
    assert sum(len(v) == 2 for v in judges.values()) == 10
    assert all(len(set(v)) == len(v) for v in judges.values())


def test_kappa_and_consolidation():
    assert weighted_kappa([(0, 0), (1, 1), (2, 2), (2, 2)]) == 1.0
    assert weighted_kappa([(0, 2), (2, 0)]) == pytest.approx(-1.0)
    assert 0 < weighted_kappa([(0, 0), (1, 1), (1, 2), (2, 2)]) < 1
    raw = {("q", "a"): {"A": 2, "B": 1}, ("q", "b"): {"A": 0}}
    assert consolidate(raw) == {"q": {"a": 1, "b": 0}}                           # lower grade wins
    assert consolidate(raw, {("q", "a"): 2})["q"]["a"] == 2                      # adjudication overrides
