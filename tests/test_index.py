import math
import random
import re

import pytest

from trademesh.index import PositionalIndex, TfidfIndex, WildcardIndex, BooleanRetriever, build_retrievers, intersect
from trademesh.types import Passage


def P(i, text):
    return Passage(id=i, ticker="X", company="X", fiscal_year=2025, item="1A", kind="text", text=text)


DOCS = [P("d0", "The company faces tariffs on imported steel and aluminum."),
        P("d1", "Supply chain disruption may delay shipments of steel."),
        P("d2", "Cyber security incidents and cyber attacks could harm the company."),
        P("d3", "Tariffs and trade restrictions raise costs; supply chain risk is high."),
        P("d4", "Dividends were paid quarterly.")]


def ids(ix, docs):
    return [ix.ids[d] for d in docs]


# ---------------------------------------------------------------- skip pointers

def test_intersect_matches_set_intersection_and_skips_save_comparisons():
    rng = random.Random(3)
    for _ in range(50):
        a = sorted(rng.sample(range(5000), rng.randint(0, 400)))
        b = sorted(rng.sample(range(5000), rng.randint(0, 40)))
        assert intersect(a, b, True)[0] == intersect(a, b, False)[0] == sorted(set(a) & set(b))
    big, small = list(range(0, 10000, 2)), [4000, 4002, 9000]
    r1, c_skip = intersect(big, small, True)
    r2, c_plain = intersect(big, small, False)
    assert r1 == r2 == small and c_skip < c_plain / 5            # long list vs short list: skips pay off


# ---------------------------------------------------------------- Boolean, phrase, proximity

def test_boolean_queries():
    ix = PositionalIndex(DOCS)
    assert ids(ix, ix.boolean("tariffs AND steel")) == ["d0"]
    assert ids(ix, ix.boolean("tariffs steel")) == ["d0"]                              # implicit AND
    assert ids(ix, ix.boolean("tariffs OR dividends")) == ["d0", "d3", "d4"]
    assert ids(ix, ix.boolean("steel AND NOT tariffs")) == ["d1"]
    assert ids(ix, ix.boolean("(tariffs OR cyber) AND company")) == ["d0", "d2"]
    assert ids(ix, ix.boolean("NOT company AND steel")) == ["d1"]
    assert ix.boolean("nonexistent") == []
    with pytest.raises(ValueError):
        ix.boolean("(tariffs")


def test_phrase_and_proximity():
    ix = PositionalIndex(DOCS)
    assert ids(ix, ix.boolean('"supply chain"')) == ["d1", "d3"]
    assert ids(ix, ix.boolean('"chain supply"')) == []                                  # order matters
    assert ids(ix, ix.boolean('"cyber attacks"')) == ["d2"]
    assert ids(ix, ix.boolean('"the company"')) == ["d0", "d2"]                         # stop words are indexed, so this works
    assert ids(ix, ix.near(["tariffs", "steel"], 3)) == ["d0"]                          # "tariffs on imported steel": 3 apart
    assert ids(ix, ix.near(["tariffs", "steel"], 2)) == []
    assert ix.phrase(["a", "nonword"]) == []


def test_numbers_match_with_or_without_commas():
    ix = PositionalIndex([P("n", "Net sales were $383,285 million.")])
    assert ix.boolean("383285") == [0] and ix.boolean("383,285") == [0]


# ---------------------------------------------------------------- wildcard

def test_wildcards_permuterm_and_kgram_agree_with_brute_force():
    terms = ["cyber", "cybersecurity", "cyclical", "cycle", "tariff", "tariffs", "cost", "costs", "coast", "co-op", "x"]
    w = WildcardIndex(terms)
    for pat in ["cyber*", "*s", "c*t", "*ost*", "ta*s", "c*c*e", "*", "cyc*l", "nomatch*", "co*"]:
        brute = sorted(t for t in terms if re.fullmatch(".*".join(map(re.escape, pat.split("*"))), t))
        assert w.kgram(pat) == brute, pat
        if pat.count("*") == 1:
            assert w.permuterm(pat) == brute, pat
        assert w.expand(pat) == brute, pat
    assert w.expand("cost") == ["cost"] and w.expand("zzz") == []
    ix = PositionalIndex(DOCS)
    assert ids(ix, ix.boolean("cyber*")) == ["d2"]
    assert ids(ix, ix.boolean("tariff*")) == ["d0", "d3"]


# ---------------------------------------------------------------- TF-IDF (hand-checked)

def test_tfidf_matches_hand_computation():
    docs = [P("a", "apple apple banana"), P("b", "banana cherry"), P("c", "cherry cherry cherry")]
    ix = TfidfIndex(docs)
    # N=3. idf(apple)=log10(3/1)=0.4771, banana=log10(3/2)=0.1761, cherry=0.1761
    # doc a: apple (1+log10 2)*0.4771 = 0.6207 ; banana 0.1761 ; norm = sqrt(0.6207^2+0.1761^2)=0.6452
    # query "apple": cosine = 0.6207/0.6452 = 0.9620
    hits = ix.search("apple")
    assert [h.passage_id for h in hits] == ["a"]
    assert hits[0].score == pytest.approx(0.6207 / math.hypot(0.6207, 0.1761), abs=1e-3)
    assert ix.search("banana")[0].passage_id in ("a", "b")
    assert ix.search("zzz") == [] and ix.search("") == []


def test_tfidf_ranks_by_tf_then_normalises_length():
    docs = [P("short", "tariff tariff"), P("long", "tariff " + "filler " * 30 + "words"), P("none", "dividend")]
    assert [h.passage_id for h in TfidfIndex(docs).search("tariff")] == ["short", "long"]


def test_champion_lists_restrict_candidates_and_fall_back():
    docs = [P(f"d{i}", ("tariff " * (i % 7 + 1)) + f"filler{i}") for i in range(60)]
    full, champ = TfidfIndex(docs), TfidfIndex(docs, champions=5)
    full.search("tariff", 5); champ.search("tariff", 5)
    assert champ.last_candidates == 5 < full.last_candidates == 60
    assert [h.passage_id for h in champ.search("tariff", 5)][0] in {h.passage_id for h in full.search("tariff", 8)}
    champ.search("tariff", 30)                                # more wanted than champions hold -> full scan
    assert champ.fallbacks == 1 and champ.last_candidates == 60


# ---------------------------------------------------------------- retrievers and contract

def test_boolean_retriever_semantics():
    b = BooleanRetriever(PositionalIndex(DOCS))
    assert [h.passage_id for h in b.search("tariffs steel")] == ["d0"]
    assert b.search("tariffs dividends") == []                                         # no passage has both: Boolean returns nothing
    assert [h.passage_id for h in b.search('"supply chain" AND tariffs')] == ["d3"]
    assert [h.passage_id for h in b.search("cyber*")] == ["d2"]


def test_build_retrievers_contract(passages):
    systems = build_retrievers(passages)
    assert set(systems) == {"tfidf", "tfidf-champ-50", "tfidf-champ-200", "boolean"}
    for s in systems.values():
        assert isinstance(s.search("tariffs", 5), list)
