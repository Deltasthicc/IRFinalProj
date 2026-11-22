"""The human-rating tooling: blind package, the rating app, and scoring. All on small synthetic data."""
import csv
import json
from pathlib import Path

import pytest

from eval import human_prepare as hp
from eval import human_score as hs
from trademesh.types import Passage

APP = Path(__file__).resolve().parents[1] / "eval" / "human_app.py"
CONFIGS = ["closed_book", "bm25_rag", "best_rag"]


def passages():
    return {f"p{i}": Passage(f"p{i}", "AAPL", "Apple", 2025, "1A", "text", f"passage text {i} about supplier risk") for i in range(1, 7)}


def runs(n=4):
    out = []
    for i in range(1, n + 1):
        out.append({"id": f"Q{i}", "question": f"question {i}?", "type": "entity" if i % 2 else "topical", "needs_table": "no",
                    "closed_book": {"answer": f"closed answer {i}", "cited": [], "passage_ids": [], "verifier_passed": None, "n_numbers": None, "numbers_grounded": None},
                    "bm25_rag": {"answer": f"bm25 answer {i} [P1]", "cited": [1], "passage_ids": ["p1", "p2"], "verifier_passed": True, "n_numbers": 1, "numbers_grounded": True},
                    "best_rag": {"answer": f"best answer {i} [P1][P2]", "cited": [1, 2], "passage_ids": ["p3", "p4"], "verifier_passed": True, "n_numbers": 0, "numbers_grounded": True}})
    return {"split": "test", "reportable": True, "runs": out}


QRELS = [{"query_id": "Q1", "passage_id": "p5", "rel": "2", "judge": "AI"}, {"query_id": "Q1", "passage_id": "p6", "rel": "1", "judge": "AI"},
         {"query_id": "Q2", "passage_id": "p5", "rel": "0", "judge": "AI"}]


# ---------------------------------------------------------------- the blind package

def test_package_is_blind_complete_and_deterministic():
    blind, key = hp.build_package(runs(), passages(), QRELS, seed=5)
    text = json.dumps(blind)
    for banned in ("closed_book", "bm25_rag", "best_rag", "closed answer 1 [P"):
        assert banned not in text.replace("closed answer", "")           # no system names anywhere in what the rater sees
    for q in blind["questions"]:
        assert sorted(q["answers"]) == ["A", "B", "C"] and sorted(key[q["id"]].values()) == sorted(CONFIGS)
        for label, a in q["answers"].items():                              # each label shows the answer of the system the key says
            cfg = key[q["id"]][label]
            assert a["text"].startswith({"closed_book": "closed", "bm25_rag": "bm25", "best_rag": "best"}[cfg])
        assert set(q["answers"]["A"]) == {"text", "cited"} and set(q) == {"id", "question", "type", "answers", "hints"}
    assert hp.build_package(runs(), passages(), QRELS, seed=5) == (blind, key)         # same seed, same labels
    assert hp.build_package(runs(8), passages(), QRELS, seed=6)[1] != hp.build_package(runs(8), passages(), QRELS, seed=7)[1]
    q1 = blind["questions"][0]
    assert [h["text"] for h in q1["hints"]] == ["passage text 5 about supplier risk"]   # only grade-2 passages are hints
    assert [c["ref"] for c in q1["answers"][next(l for l, c in key["Q1"].items() if c == "best_rag")]["cited"]] == ["P1", "P2"]


def test_retrieval_sample_is_stratified_and_test_only():
    needs = ([{"id": f"E{i:02d}", "type": "entity", "split": "test"} for i in range(1, 8)] +
             [{"id": f"T{i:02d}", "type": "topical", "split": "test"} for i in range(1, 8)] +
             [{"id": f"R{i:02d}", "type": "terminology", "split": "test"} for i in range(1, 7)] + [{"id": "E99", "type": "entity", "split": "dev"}])
    s = hp.retrieval_sample(needs)
    assert len(s) == 8 and "E99" not in s and sum(x.startswith("E") for x in s) == 3 and sum(x.startswith("R") for x in s) == 2
    assert s == hp.retrieval_sample(needs)


def test_retrieval_sheets_are_blank_copies(monkeypatch, tmp_path):
    src = tmp_path / "judg"
    src.mkdir()
    fields = ["query_id", "need", "passage_id", "ticker", "fiscal_year", "item", "kind", "passage_text", "rel", "judge"]
    with open(src / "pool_sheet_AI.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for q in ("E01", "E02"):
            w.writerow({"query_id": q, "need": "n", "passage_id": "p1", "ticker": "AAPL", "fiscal_year": 2025, "item": "1A", "kind": "text",
                        "passage_text": "t", "rel": "2", "judge": "AI"})
    monkeypatch.setattr(hp, "JUDG", src)
    assert hp.write_retrieval_sheets(["Sam"], ["E01"], tmp_path / "out") == 1
    rows = list(csv.DictReader(open(tmp_path / "out" / "pool_sheet_Sam.csv", encoding="utf-8")))
    assert rows[0]["rel"] == "" and rows[0]["judge"] == "Sam" and rows[0]["query_id"] == "E01"      # the model's grade is not copied


def test_the_committed_package_matches_the_study_and_reshuffles_the_labels():
    root = Path(hp.HUMAN)
    if not (root / "blind_answers.json").exists():
        pytest.skip("package not generated")
    blind = json.loads((root / "blind_answers.json").read_text(encoding="utf-8"))
    key = json.loads((root / "key_DO_NOT_OPEN.json").read_text(encoding="utf-8"))
    llm_key = json.loads((hp.OUT / "rating_key.json").read_text(encoding="utf-8"))
    assert len(blind["questions"]) == 20 and sum(len(q["answers"]) for q in blind["questions"]) == 60
    assert sum(len(a["cited"]) for q in blind["questions"] for a in q["answers"].values()) == len(hp.read_csv(hp.OUT / "citation_sheet.csv"))
    assert sum(key[q][l] != llm_key[q][l] for q in key for l in "ABC") > 20                 # a different order from the model-rated sheets
    for q in blind["questions"]:
        assert sorted(key[q["id"]].values()) == sorted(CONFIGS)


# ---------------------------------------------------------------- scoring

def _setup_scoring(tmp_path, monkeypatch, ratings_by_rater, complete=True):
    human = tmp_path / "human"
    (human / "raters").mkdir(parents=True)
    blind, key = hp.build_package(runs(), passages(), QRELS, seed=1)
    (human / "blind_answers.json").write_text(json.dumps(blind), encoding="utf-8")
    (human / "key_DO_NOT_OPEN.json").write_text(json.dumps(key), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "rag_runs_test.json").write_text(json.dumps(runs()), encoding="utf-8")
    cfg_of = {(q, l): c for q, m in key.items() for l, c in m.items()}
    for name, score in ratings_by_rater.items():
        rdir = human / "raters" / name
        rdir.mkdir()
        rows = [{"question_id": q, "label": l, "rating": str(score[c])} for (q, l), c in cfg_of.items()]
        rows = rows if complete else rows[:5]
        with open(rdir / "ratings.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["question_id", "label", "rating"])
            w.writeheader()
            w.writerows(rows)
        crow = [{"question_id": q["id"], "label": l, "citation": c["ref"], "supports": "1"} for q in blind["questions"] for l, a in q["answers"].items() for c in a["cited"]]
        crow[0]["supports"] = "0"
        with open(rdir / "citations.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["question_id", "label", "citation", "supports"])
            w.writeheader()
            w.writerows(crow)
    saved = {}
    monkeypatch.setattr(hs, "HUMAN", human)
    monkeypatch.setattr(hs, "OUT", out)
    monkeypatch.setattr(hs, "ROOT", tmp_path)
    (tmp_path / "docs" / "report_C").mkdir(parents=True)
    monkeypatch.setattr(hs, "write_json", lambda name, obj, dry=False: saved.setdefault("report", obj) and (out / name))
    return saved


def test_scoring_uses_only_the_human_ratings(tmp_path, monkeypatch, capsys):
    saved = _setup_scoring(tmp_path, monkeypatch, {"Sam": {"closed_book": 0, "bm25_rag": 1, "best_rag": 2}})
    monkeypatch.setattr("sys.argv", ["human_score.py"])
    hs.main()
    rep = saved["report"]
    assert rep["rated_by"] == ["Sam"] and rep["n_questions"] == 4
    assert [rep["configs"][c]["mean_rating"] for c in CONFIGS] == [0.0, 1.0, 2.0]
    assert rep["configs"]["best_rag"]["share_rated_2"] == 1.0 and rep["configs"]["closed_book"]["share_rated_2"] == 0.0
    assert rep["configs"]["closed_book"]["citation_precision"] is None                   # no citations to judge
    assert rep["configs"]["bm25_rag"]["number_grounding"] == 1.0                         # automatic, from the verifier
    assert rep["bootstrap"]["best_rag - closed_book"]["mean_diff"] == 2.0 and "kappa_between_humans" not in rep
    assert (tmp_path / "docs" / "report_C" / "human_rating_results.md").read_text(encoding="utf-8").startswith("# RAG study: human ratings")


def test_two_raters_give_a_kappa_and_the_citation_rule_is_strict(tmp_path, monkeypatch):
    saved = _setup_scoring(tmp_path, monkeypatch, {"Sam": {"closed_book": 0, "bm25_rag": 1, "best_rag": 2},
                                                   "Alex": {"closed_book": 0, "bm25_rag": 2, "best_rag": 2}})
    monkeypatch.setattr("sys.argv", ["human_score.py", "--no-md"])
    hs.main()
    rep = saved["report"]
    assert rep["n_double_rated"] == 12 and 0 < rep["kappa_between_humans"] < 1
    assert [rep["configs"][c]["mean_rating"] for c in CONFIGS] == [0.0, 1.5, 2.0]        # mean over raters
    cp = [rep["configs"][c]["n_citations_judged"] for c in CONFIGS]
    assert cp == [0, 4, 8]
    assert rep["per_rater"]["Sam"] == {"closed_book": 0.0, "bm25_rag": 1.0, "best_rag": 2.0}
    assert rep["per_rater"]["Alex"] == {"closed_book": 0.0, "bm25_rag": 2.0, "best_rag": 2.0}


def test_incomplete_sheets_are_refused(tmp_path, monkeypatch):
    _setup_scoring(tmp_path, monkeypatch, {"Sam": {"closed_book": 0, "bm25_rag": 1, "best_rag": 2}}, complete=False)
    monkeypatch.setattr("sys.argv", ["human_score.py"])
    with pytest.raises(SystemExit) as e:
        hs.main()
    assert "incomplete" in str(e.value)


def test_no_ratings_means_no_scores(tmp_path, monkeypatch):
    _setup_scoring(tmp_path, monkeypatch, {})
    monkeypatch.setattr("sys.argv", ["human_score.py"])
    with pytest.raises(SystemExit) as e:
        hs.main()
    assert "no human ratings" in str(e.value)


def test_retrieval_validation_compares_grades(tmp_path, monkeypatch):
    human = tmp_path / "human" / "judgments"
    human.mkdir(parents=True)
    judg = tmp_path / "judg"
    judg.mkdir()
    fields = ["query_id", "need", "passage_id", "ticker", "fiscal_year", "item", "kind", "passage_text", "rel", "judge"]

    def sheet(path, grades, judge):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i, g in enumerate(grades):
                w.writerow({"query_id": "E01", "need": "n", "passage_id": f"p{i}", "ticker": "A", "fiscal_year": 2025, "item": "1", "kind": "text",
                            "passage_text": "t", "rel": g, "judge": judge})
    sheet(human / "pool_sheet_Sam.csv", ["2", "0", "1", "2"], "Sam")
    sheet(judg / "pool_sheet_AI.csv", ["2", "0", "2", "2"], "AI")
    monkeypatch.setattr(hs, "HUMAN", tmp_path / "human")
    monkeypatch.setattr(hs, "JUDG", judg)
    v = hs.retrieval_validation(["Sam"])["Sam vs AI"]
    assert v["n"] == 4 and v["identical"] == 0.75 and v["same_relevant_or_not"] == 1.0 and v["confusion"]["human 1 / model 2"] == 1
    assert hs.retrieval_validation(["Nobody"]) is None


# ---------------------------------------------------------------- the rating app

def test_rating_app_saves_every_step_and_never_shows_a_system_name(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    blind, key = hp.build_package(runs(), passages(), QRELS, seed=3)
    (tmp_path / "blind_answers.json").write_text(json.dumps(blind), encoding="utf-8")
    monkeypatch.setenv("TRADEMESH_HUMAN", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception and any("name" in i.value.lower() for i in at.info)            # asks for a name first
    at.sidebar.text_input(key="rater").set_value("Sam").run()
    assert not at.exception
    # step 1: reference answer (inside a form)
    at.text_area[0].set_value("The company says it depends on key people.")
    next(b for b in at.button if "Save reference" in b.label).click().run()
    assert (tmp_path / "raters" / "Sam" / "references.csv").read_text(encoding="utf-8").count("depends on key people") == 1
    # step 2: rate the three answers of the first question
    at.sidebar.radio(key="step").set_value("2. Rate answers").run()
    shown = " ".join(m.value for m in at.markdown)
    assert "Answer A" in shown and "Answer B" in shown and "closed_book" not in shown and "best_rag" not in shown
    for label, score in zip("ABC", ("2", "1", "0")):
        at.radio(key=f"rate_Q1_{label}").set_value(score).run()
    rows = list(csv.DictReader(open(tmp_path / "raters" / "Sam" / "ratings.csv", encoding="utf-8")))
    assert {(r["question_id"], r["label"], r["rating"]) for r in rows} == {("Q1", "A", "2"), ("Q1", "B", "1"), ("Q1", "C", "0")}
    # step 3: citations
    at.sidebar.radio(key="step").set_value("3. Check citations").run()
    cit_keys = [r.key for r in at.radio if r.key and r.key.startswith("cit_Q1_")]
    assert len(cit_keys) == 3                                               # bm25 cites 1, best cites 2
    at.radio(key=cit_keys[0]).set_value("1").run()
    assert not at.exception
    assert len(list(csv.DictReader(open(tmp_path / "raters" / "Sam" / "citations.csv", encoding="utf-8")))) == 1


def test_leave_one_out_and_rq3_helpers():
    per = {"closed_book": {"a": 0, "b": 0, "c": 0}, "bm25_rag": {"a": 1, "b": 1, "c": 1}, "best_rag": {"a": 2, "b": 2, "c": 0}}
    loo = hs.leave_one_out(per, ["a", "b", "c"])
    assert loo["best_rag - closed_book"] == {"min": 1.0, "max": 2.0}                 # dropping c gives 2.0, dropping a or b gives 1.0
    assert loo["bm25_rag - closed_book"] == {"min": 1.0, "max": 1.0}
    pairs = [[0.1, 9], [0.4, 9], [0.7, 9], [0.9, 9], [0.2, 9], [0.5, 9]]               # the stored ratings (9) are ignored
    order = [("q1", "bm25_rag"), ("q2", "bm25_rag"), ("q3", "bm25_rag"), ("q1", "best_rag"), ("q2", "best_rag"), ("q3", "best_rag")]
    human = {k: v for k, v in zip(order, [0, 1, 1, 2, 0, 2])}
    r = hs.rq3_with_human(pairs, order, human)
    assert r["n"] == 6 and -1 <= r["spearman"] <= 1 and r["ci95_approx"][0] < r["spearman"] < r["ci95_approx"][1]
    perfect = hs.rq3_with_human(pairs, order, {k: p[0] * 10 for k, p in zip(order, pairs)})
    assert abs(perfect["spearman"] - 1) < 1e-9
