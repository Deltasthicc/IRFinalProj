"""End-to-end smoke test of Role B's evaluation scripts on the TEST FIXTURE.

The judgments below are SYNTHETIC (a word-overlap rule written in this test) and exist only to check
that the machinery runs from pooling to the final tables. They are never written to data/judgments and
nothing here is a result. Real judgments come from people (eval/make_pools.py).
"""
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script, env, *args, check=True):
    p = subprocess.run([sys.executable, f"eval/{script}.py", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if check:
        assert p.returncode == 0, p.stdout + p.stderr
    return p


def words(t):
    return set(re.findall(r"[a-z]+", t.lower()))


def test_pipeline_refuses_then_runs(tmp_path):
    env = {**os.environ, "TRADEMESH_JUDGMENTS": str(tmp_path), "PYTHONPATH": str(ROOT / "src")}
    # 1) without --allow-stub the fixture is refused
    p = run("run_retrieval", env, check=False)
    assert p.returncode != 0 and "Refusing" in (p.stdout + p.stderr)
    # 2) pooling writes blind sheets (dry run -> dryrun/ subfolder)
    run("make_pools", env, "--allow-stub", "--judges", "A,B,C")
    sheets = sorted((tmp_path / "dryrun").glob("pool_sheet_*.csv"))
    assert sheets and (tmp_path / "dryrun" / "pool_key.json").exists()
    header = sheets[0].read_text().splitlines()[0]
    assert "bm25" not in header and "rank" not in header.lower() and "system" not in header.lower()
    # 3) judge synthetically, merge, and run every analysis script
    for s in sheets:
        rows = list(csv.DictReader(open(s, encoding="utf-8", newline="")))
        for r in rows:
            ov = len(words(r["need"]) & words(r["passage_text"]))
            r["rel"] = "2" if ov >= 4 else "1" if ov >= 2 else "0"
        with open(tmp_path / s.name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (tmp_path / "pool_key.json").write_text((tmp_path / "dryrun" / "pool_key.json").read_text())
    out = run("make_pools", env, "--merge").stdout
    assert "merged" in out and "0 still blank" in out
    res = run("run_retrieval", env, "--allow-stub", "--split", "dev").stdout
    assert "nDCG@10" in res and "bm25" in res and "bm25+prf" in res and "DRY RUN" in res
    assert "paired bootstrap" in res
    data = json.loads((ROOT / "eval/out/dryrun/retrieval_dev.json").read_text())
    assert 0 <= data["overall"]["bm25"]["nDCG@10"] <= 1
    assert "best: k1=" in run("tune_bm25", env, "--allow-stub").stdout
    assert not (tmp_path / "bm25_params.json").exists()          # a dry run must not write tuned parameters
    assert "A prep" in run("ablation", env, "--allow-stub", "--split", "dev").stdout
    assert "Leave-one-system-out" in run("pool_bias", env, "--allow-stub").stdout
    assert "p50" in run("cost_table", env, "--allow-stub").stdout


def test_judge_app_saves_grades(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    sheet = tmp_path / "pool_sheet_A.csv"
    with open(sheet, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "need", "passage_id", "ticker", "fiscal_year", "item", "kind", "passage_text", "rel", "judge"])
        w.writerow(["E01", "a need", "p1", "AAPL", "2025", "1A", "text", "some text", "", "A"])
        w.writerow(["E01", "a need", "p2", "AAPL", "2025", "7", "text", "other text", "", "A"])
    monkeypatch.setenv("TRADEMESH_JUDGMENTS", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "eval" / "judge_app.py"), default_timeout=20).run()
    assert not at.exception
    at.radio[0].set_value("2").run()
    assert not at.exception
    rows = list(csv.DictReader(open(sheet, encoding="utf-8", newline="")))
    assert [r["rel"] for r in rows] == ["2", ""]


def test_hypothesis_rules_apply_mechanically():
    sys.path.insert(0, str(ROOT))
    from eval.hypotheses import evaluate
    ids = [f"q{i}" for i in range(12)]
    types = {q: ("entity" if i < 4 else "topical" if i < 8 else "terminology") for i, q in enumerate(ids)}
    flat = {q: {"nDCG@10": 0.5} for q in ids}
    better = {q: {"nDCG@10": 0.8 + 0.01 * i} for i, q in enumerate(ids)}          # synthetic, rule check only
    res = {h: r for h, _, r in evaluate({"bm25": flat, "tfidf": better, "dense": flat, "hybrid": better,
                                        "hybrid+rerank": better, "bm25+prf": flat}, types)}
    assert res["H1"]["verdict"] == "B>A (A REFUTED)"          # bm25 (flat 0.5) is clearly worse than tfidf (0.8+)
    assert res["H3a"]["verdict"] == "A>B SUPPORTED"
    assert res["H4"]["verdict"].startswith("TIED")            # identical systems
    assert res["H6a"]["mean_diff"] == 0
