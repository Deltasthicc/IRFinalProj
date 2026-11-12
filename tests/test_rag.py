import json

import pytest

from trademesh.dev_stub import Bm25Stub
from trademesh.query.entities import EntityMatcher
from trademesh.query.section_prior import apply_prior, g, ndcg_at_k, tune_lambda
from trademesh.rag.abstain import abstention_metrics, confidence, decide, tune_tau
from trademesh.rag.llm import CacheMiss, LiveCallCapReached, LLMClient
from trademesh.rag.pipeline import RagConfig, RagPipeline
from trademesh.rag.prompt import build_context, is_insufficient
from trademesh.rag.verifier import extract_numbers, verify
from trademesh.types import Hit, Passage


def P(i, text, year=2025, item="7", ticker="AAPL"):
    return Passage(f"id{i}", ticker, "Co", year, item, "text", text)


# ---------------------------------------------------------------- prompt / context

def test_context_labels_and_metadata():
    ctx = build_context([P(1, "Alpha."), P(2, "Beta.", item="1A", ticker="MSFT")])
    assert "[P1] AAPL | FY2025 | Item 7 | text\nAlpha." in ctx
    assert "[P2] MSFT | FY2025 | Item 1A | text\nBeta." in ctx


def test_is_insufficient():
    assert is_insufficient("INSUFFICIENT EVIDENCE")
    assert is_insufficient("  insufficient evidence. ")
    assert not is_insufficient("There is insufficient evidence for X, but Y [P1].")


# ---------------------------------------------------------------- verifier

def test_extract_numbers_normalisation():
    vals = dict(extract_numbers("Revenue was $1,200 million, up 4% to $1.2 billion; see Item 1A of the 10-K [P2]."))
    assert 1.2e9 in vals.values() and 4.0 in vals.values()
    raws = [r for r, _ in extract_numbers("Item 1A of the 10-K [P2]")]
    assert raws == []                       # Item numbers, form name and labels are not claims


def test_verifier_pass():
    ps = [P(1, "Net sales increased 4 percent to $400,000 million in 2025.")]
    r = verify("Net sales rose 4% to $400 billion [P1].", ps)
    assert r.passed and r.cited == [1] and not r.ungrounded


def test_verifier_flags_invented_citation_and_number():
    ps = [P(1, "Net sales increased 4 percent.")]
    r = verify("Net sales rose 7 percent [P1]. Margin was 40% [P3].", ps)
    assert r.invalid_cites == [3]
    assert [n.raw for n in r.ungrounded] == ["7", "40"] or {n.raw for n in r.ungrounded} == {"7", "40"}
    assert not r.passed and len(r.flags) >= 2


def test_citation_after_final_period_belongs_to_that_sentence():
    r = verify("Net sales rose. [P1]", [P(1, "Net sales rose.")])
    assert r.passed and r.uncited_sentences == []
    r = verify("Net sales rose [P1]. Margins fell.", [P(1, "Net sales rose.")])
    assert r.uncited_sentences == ["Margins fell."]


def test_verifier_no_citation():
    r = verify("Net sales rose.", [P(1, "Net sales rose.")])
    assert "answer contains no citation" in r.flags


def test_verifier_number_only_in_uncited_passage():
    ps = [P(1, "Alpha grew 5 percent."), P(2, "Beta grew 9 percent.")]
    r = verify("Alpha grew 9 percent [P1].", ps)
    assert r.numbers[0].grounded_in == [] and r.numbers[0].in_uncited == [2]
    assert not r.passed


def test_verifier_table_scale_hint():
    ps = [P(1, "Total net sales (in millions). Total 383,285.")]
    assert verify("Total net sales were 383,285 million [P1].", ps).passed
    assert verify("Total net sales were $383.285 billion [P1].", ps).passed
    assert not verify("Total net sales were $383.3 billion [P1].", ps).passed   # rounding is flagged


def test_verifier_abstention_makes_no_claims():
    assert verify("INSUFFICIENT EVIDENCE", []).passed


def test_verifier_fiscal_year_in_header_counts():
    assert verify("In 2025 sales rose [P1].", [P(1, "Sales rose.")]).passed


# ---------------------------------------------------------------- abstention

def test_confidence_and_decide():
    assert confidence([], "q") == 0.0
    assert confidence([Hit("a", 6.0)], "supplier concentration risk") == 2.0   # 3 terms, 'risk' kept
    assert decide("ok [P1].", 0.5, 1.0).abstain
    assert decide("INSUFFICIENT EVIDENCE", 5.0, 1.0).abstain
    assert not decide("ok [P1].", 5.0, 1.0).abstain


def test_abstention_metrics():
    m = abstention_metrics([(True, True), (True, False), (False, False), (False, True)])
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)
    assert m.precision == m.recall == m.accuracy == 0.5


def test_tune_tau_separates_classes():
    items = [(0.2, True, False), (0.3, True, False), (1.0, False, False), (1.5, False, False)]
    tau, m = tune_tau(items)
    assert 0.3 < tau <= 1.0 and m.f1 == 1.0


# ---------------------------------------------------------------- LLM client / cache

def test_llm_cache_and_cap(tmp_path):
    c = LLMClient(provider="stub", cache_dir=tmp_path, max_live_calls=1)
    user = "PASSAGES\n[P1] AAPL | FY2025 | Item 7 | text\nSales rose 4 percent. More.\n\nQUESTION: q"
    a = c.complete("sys", user)
    assert not a.cached and a.text == "Sales rose 4 percent. [P1]"
    b = c.complete("sys", user)
    assert b.cached and b.text == a.text and c.live_calls == 1 and c.cache_hits == 1
    with pytest.raises(LiveCallCapReached):
        c.complete("sys", user + " different")
    log = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["cached"] for r in log] == [False, True]


def test_llm_offline_cache_miss(tmp_path):
    c = LLMClient(provider="stub", cache_dir=tmp_path, offline=True)
    with pytest.raises(CacheMiss):
        c.complete("s", "u")


def test_llm_cache_key_depends_on_model(tmp_path):
    a = LLMClient(provider="stub", model="m1", cache_dir=tmp_path)
    b = LLMClient(provider="stub", model="m2", cache_dir=tmp_path)
    assert a._key("s", "u", 500) != b._key("s", "u", 500)


# ---------------------------------------------------------------- section prior

def test_section_prior():
    by_id = {"a": P(1, "x", item="9"), "b": P(2, "y", item="1A")}
    by_id["c"] = P(3, "z", item="9")
    hits = [Hit("a", 10.0), Hit("b", 9.5), Hit("c", 0.0)]
    assert apply_prior(hits, by_id, 0.0) == hits
    out = apply_prior(hits, by_id, 0.5)
    assert [h.passage_id for h in out][:2] == ["b", "a"]      # Item 1A (g=1) overtakes Item 9 (g=0.1)
    assert g("1a") == 1.0 and g("zz") == 0.3


def test_ndcg_and_tune_lambda():
    assert ndcg_at_k(["a", "b"], {"a"}, 10) == 1.0
    assert 0 < ndcg_at_k(["x", "a"], {"a"}, 10) < 1
    by_id = {"a": P(1, "x", item="9"), "b": P(2, "y", item="1A")}
    by_id["c"] = P(3, "z", item="9")
    search = lambda q: [Hit("a", 10.0), Hit("b", 9.5), Hit("c", 0.0)]
    lam, scores = tune_lambda(search, by_id, [("q", {"b"})])
    assert lam > 0 and scores[lam] > scores[0.0]


# ---------------------------------------------------------------- pipeline

@pytest.fixture()
def pipe(passages, companies, tmp_path):
    llm = LLMClient(provider="stub", cache_dir=tmp_path)
    vocab = {w for p in passages for w in p.text.lower().split()}
    return RagPipeline(passages, Bm25Stub(passages), llm, EntityMatcher(companies, known_words=vocab))


def test_single_company_answer_is_filtered_and_verified(pipe):
    r = pipe.answer("What does Tesla say about supplier concentration?")
    assert r.mode == "rag" and [m.ticker for m in r.mentions] == ["TSLA"]
    assert r.passages and all(p.ticker == "TSLA" for p in r.passages)
    assert "tesla" not in r.topic_query.lower()
    assert r.verifier is not None and r.verifier.cited == [1] and not r.abstained


def test_typo_is_corrected_and_reported(pipe):
    r = pipe.answer("Telsa supplier concentration")
    assert r.corrections == [("Telsa", "Tesla, Inc.")]
    assert all(p.ticker == "TSLA" for p in r.passages)


def test_compare_mode_groups_passages_by_company(pipe):
    r = pipe.answer("Compare Apple and Microsoft on supplier concentration")
    assert r.mode == "compare" and set(r.by_company) == {"AAPL", "MSFT"}
    assert len(r.passages) == 6
    for t, labels in r.by_company.items():
        assert all(r.passages[i - 1].ticker == t for i in labels)


def test_closed_book_has_no_passages(pipe):
    r = pipe.answer("What was Apple's revenue?", closed_book=True)
    assert r.mode == "closed_book" and r.passages == [] and r.verifier is None


def test_low_confidence_abstains_without_llm_call(passages, companies, tmp_path):
    llm = LLMClient(provider="stub", cache_dir=tmp_path)
    pipe = RagPipeline(passages, Bm25Stub(passages), llm, EntityMatcher(companies), config=RagConfig(tau=999.0))
    r = pipe.answer("supplier concentration")
    assert r.abstained and "below the threshold" in r.abstain_reason and llm.live_calls == 0


def test_no_hits_abstains(passages, companies, tmp_path):
    llm = LLMClient(provider="stub", cache_dir=tmp_path)
    pipe = RagPipeline(passages, Bm25Stub(passages), llm, EntityMatcher(companies))
    r = pipe.answer("zzzz qqqq")
    assert r.abstained and r.abstain_reason == "no passage was retrieved" and llm.live_calls == 0


def test_second_run_is_served_from_cache(pipe):
    pipe.answer("Tesla battery supplier risk")
    assert pipe.llm.live_calls == 1
    r = pipe.answer("Tesla battery supplier risk")
    assert r.cached and pipe.llm.live_calls == 1


def test_gemini_request_and_response(tmp_path, monkeypatch):
    """The Gemini adapter builds the right request and parses the reply (no network: fake requests)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Sales rose 4 percent [P1]."}]}}]}

    class FakeRequests:
        class exceptions:                      # what llm._post catches and retries on
            Timeout = TimeoutError
            ConnectionError = ConnectionError

        @staticmethod
        def post(url, headers, json, timeout):
            seen.update(url=url, headers=headers, body=json)
            return FakeResp()

    import sys
    monkeypatch.setitem(sys.modules, "requests", FakeRequests)
    monkeypatch.setenv("LLM_MIN_INTERVAL", "0")
    c = LLMClient(provider="gemini", model="gemini-3.5-flash", cache_dir=tmp_path)
    r = c.complete("be brief", "question?")
    assert r.text == "Sales rose 4 percent [P1]." and not r.cached
    assert seen["url"].endswith("/models/gemini-3.5-flash:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "test-key"
    assert seen["body"]["generationConfig"]["temperature"] == 0
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert c.complete("be brief", "question?").cached        # second call served from disk


def test_sentence_split_keeps_abbreviations_together():
    from trademesh.rag.verifier import split_sentences
    s = split_sentences("Elon Musk leads Tesla [P1]. Although Mr. Musk has other roles [P1]. "
                        "For fiscal 2026, Walmart U.S. net sales rose [P2]. Sales fell. [P3]")
    assert s == ["Elon Musk leads Tesla [P1].", "Although Mr. Musk has other roles [P1].",
                 "For fiscal 2026, Walmart U.S. net sales rose [P2].", "Sales fell. [P3]"]


def test_verifier_does_not_flag_abbreviations_but_still_flags_uncited_sentences():
    from trademesh.rag.verifier import verify
    p = Passage(id="X-1", ticker="TSLA", company="Tesla", fiscal_year=2025, item="1A", kind="text",
                text="Mr. Musk does not devote his full time to Tesla.")
    ok = verify("Although Mr. Musk has other roles, he leads Tesla [P1].", [p])
    assert ok.passed and not ok.uncited_sentences
    bad = verify("Mr. Musk leads Tesla [P1]. The company is profitable.", [p])
    assert not bad.passed and bad.uncited_sentences == ["The company is profitable."]
