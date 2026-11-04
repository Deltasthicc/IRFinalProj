from collections import Counter

from eval.common import NEEDS, read_csv
from eval.entity_eval import evaluate, load_dictionary
from eval.rag_score import cohen_kappa


def test_needs_have_the_proposed_shape():
    needs = read_csv(NEEDS / "needs.csv")
    assert len(needs) == 30
    assert Counter(n["type"] for n in needs) == {"entity": 10, "topical": 10, "terminology": 10}
    assert Counter(n["split"] for n in needs) == {"dev": 10, "test": 20}
    assert sum(n["needs_table"] == "yes" for n in needs) == 5
    assert len({n["id"] for n in needs}) == 30
    # every split keeps a mix of the three types, so tuning on dev is not blind to a type
    for split in ("dev", "test"):
        assert len({n["type"] for n in needs if n["split"] == split}) == 3


def test_unanswerable_and_typo_sets():
    assert len(read_csv(NEEDS / "unanswerable.csv")) == 10
    typos = read_csv(NEEDS / "typos.csv")
    assert len(typos) == 40
    tickers = {c["ticker"] for c in load_dictionary()[0]}
    assert {t["ticker"] for t in typos} <= tickers


def test_entity_eval_runs_and_is_consistent():
    companies, _ = load_dictionary()
    out = evaluate(companies, read_csv(NEEDS / "typos.csv"))
    assert out["n"] == 40 and out["correct"] + len(out["failures"]) == 40
    assert out["wrong_ticker"] + out["no_match"] == len(out["failures"])
    assert out["agreement_with_brute_force"] == 1.0   # the bigram shortlist loses nothing here


def test_cohen_kappa():
    assert cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    assert abs(cohen_kappa([0, 1], [1, 0]) - (-1.0)) < 1e-9
    assert 0 < cohen_kappa([0, 1, 1, 2], [0, 1, 2, 2]) < 1
