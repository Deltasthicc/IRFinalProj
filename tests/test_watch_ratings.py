"""The watcher imports another rater's complete sheets from the remote and refuses incomplete or unsafe ones."""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval import watch_ratings as wr  # noqa: E402


def git(repo, *a):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *a], check=True, capture_output=True)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


BLIND = {"questions": [{"id": "Q1", "answers": {"A": {"text": "x", "cited": [{"ref": "P1"}]}, "B": {"text": "y", "cited": []}}},
                       {"id": "Q2", "answers": {"A": {"text": "z", "cited": []}, "B": {"text": "w", "cited": [{"ref": "P2"}]}}}]}


def sheets(repo, name, ratings=None, cites=None, refs=None):
    d = repo / "eval/human/raters" / name
    write_csv(d / "references.csv", ["question_id", "reference"], refs if refs is not None else [["Q1", "r1"], ["Q2", "r2"]])
    write_csv(d / "ratings.csv", ["question_id", "label", "rating"],
              ratings if ratings is not None else [["Q1", "A", "2"], ["Q1", "B", "1"], ["Q2", "A", "0"], ["Q2", "B", "2"]])
    write_csv(d / "citations.csv", ["question_id", "label", "citation", "supports"],
              cites if cites is not None else [["Q1", "A", "P1", "1"], ["Q2", "B", "P2", "0"]])


def make_world(tmp_path):
    remote, local = tmp_path / "remote", tmp_path / "local"
    remote.mkdir()
    git(remote, "init", "-b", "main")
    (remote / "eval/human").mkdir(parents=True)
    (remote / "eval/human/blind_answers.json").write_text(json.dumps(BLIND), encoding="utf-8")
    sheets(remote, "shash")
    git(remote, "add", "-A")
    git(remote, "commit", "-m", "base")
    subprocess.run(["git", "clone", "-q", str(remote), str(local)], check=True, capture_output=True)
    return remote, local


def test_nothing_new_when_only_known_raters_exist(tmp_path):
    _, local = make_world(tmp_path)
    res = wr.check(local)
    assert res["imported"] == [] and res["incomplete"] == {} and res["known"] == ["shash"]


def test_a_complete_sheet_on_any_branch_is_imported_with_its_retrieval_sheet(tmp_path):
    remote, local = make_world(tmp_path)
    git(remote, "checkout", "-b", "ruchir/ratings")
    sheets(remote, "Ruchir", ratings=[["Q1", "A", "1"], ["Q1", "B", "1"], ["Q2", "A", "0"], ["Q2", "B", "1"]])
    write_csv(remote / "eval/human/judgments/pool_sheet_Ruchir.csv", ["query_id", "passage_id", "rel"], [["Q1", "p", "2"]])
    git(remote, "add", "-A")
    git(remote, "commit", "-m", "ratings")
    res = wr.check(local)
    assert [r["name"] for r in res["imported"]] == ["Ruchir"] and res["imported"][0]["ref"].endswith("ruchir/ratings")
    assert (local / "eval/human/raters/Ruchir/ratings.csv").exists()
    assert (local / "eval/human/judgments/pool_sheet_Ruchir.csv").exists()
    assert wr.check(local)["imported"] == []                      # idempotent: now it is a known rater


def test_incomplete_or_invalid_sheets_are_reported_and_not_imported(tmp_path):
    remote, local = make_world(tmp_path)
    sheets(remote, "Sam", ratings=[["Q1", "A", "2"], ["Q1", "B", "3"], ["Q2", "A", ""], ["Q2", "B", "2"]], refs=[["Q1", "r1"]])
    git(remote, "add", "-A")
    git(remote, "commit", "-m", "partial")
    res = wr.check(local)
    assert res["imported"] == [] and "Sam" in res["incomplete"]
    text = " ".join(res["incomplete"]["Sam"]["problems"])
    assert "not 0, 1 or 2" in text and "1 questions have no reference" in text and "answers rated" in text
    assert not (local / "eval/human/raters/Sam").exists()


def test_unsafe_folder_names_are_ignored(tmp_path):
    remote, local = make_world(tmp_path)
    sheets(remote, "bad name", )
    git(remote, "add", "-A")
    git(remote, "commit", "-m", "x")
    assert wr.check(local)["imported"] == [] and not (local / "eval/human/raters/bad name").exists()
