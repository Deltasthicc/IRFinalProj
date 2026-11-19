"""Blind HUMAN rating tool for the RAG study.   Run:  streamlit run eval/human_app.py

Four steps, each saves to eval/human/raters/<your name>/ after every click, so you can stop and resume:
  1. Write a short reference answer for each of the 20 test questions (from the filings, in your own words).
  2. Rate the 60 anonymous answers 0 / 1 / 2 against your reference and the filing.
  3. Check the 95 citations: does the cited passage support the sentence it is cited for?
  4. (Optional) Validate retrieval: grade passages 0 / 1 / 2 with eval/judge_app.py (instructions in the sidebar).

You never see which system wrote an answer, and none of the model's earlier ratings are shown.
Do NOT open eval/human/key_DO_NOT_OPEN.json until you have finished all steps.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
HUMAN = Path(os.environ.get("TRADEMESH_HUMAN", ROOT / "eval" / "human"))
SCALE = {"2": "2   correct and complete", "1": "1   partly correct, or incomplete", "0": "0   wrong, invented, or no answer"}
NO_ANSWER = "NO ANSWER IN THE CORPUS"


# ---------------------------------------------------------------- storage

def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def upsert(path: Path, fields: list[str], keys: tuple[str, ...], row: dict) -> None:
    rows = read_rows(path)
    for r in rows:
        if all(r[k] == row[k] for k in keys):
            r.update(row)
            break
    else:
        rows.append(row)
    write_rows(path, fields, rows)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", name.strip())


def esc_md(text: str) -> str:
    """Escape dollar signs so money amounts are not drawn as formulas."""
    return text.replace("\\$", "$").replace("$", "\\$")


@st.cache_resource
def load_blind() -> dict:
    return json.loads((HUMAN / "blind_answers.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading the filings for search (about 15 seconds, once)")
def corpus_search():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from trademesh.data import load_corpus
    from trademesh.retrievers import available_retrievers
    passages, _, _ = load_corpus()
    retrievers, _ = available_retrievers(passages)
    return {p.id: p for p in passages}, retrievers["bm25"] if "bm25" in retrievers else next(iter(retrievers.values()))


# ---------------------------------------------------------------- pages

def passage_box(p: dict, label: str = "") -> None:
    where = "Other sections" if p["item"] == "X" else f"Item {p['item']}"
    with st.container(border=True):
        st.markdown(f"**{label}{p['ticker']}**  ·  FY{p['fiscal_year']}  ·  {where}")
        st.write(esc_md(p["text"]))


def step_reference(q: dict, rdir: Path) -> None:
    path = rdir / "references.csv"
    cur = next((r["reference"] for r in read_rows(path) if r["question_id"] == q["id"]), "")
    st.info("Write the answer you would accept as correct, in 1 to 2 sentences, using only what the filings say. "
            "If the filings do not contain an answer, tick the box.")
    with st.expander("Hints: passages the retrieval judgments marked relevant (graded by a language model, so check them yourself)"):
        for p in q["hints"] or []:
            passage_box(p)
        if not q["hints"]:
            st.write("No hint passages for this question.")
    with st.expander("Search the filings yourself"):
        term = st.text_input("Search terms", key=f"s_{q['id']}")
        if term.strip():
            by_id, retr = corpus_search()
            for h in retr.search(term, k=6):
                p = by_id[h.passage_id]
                passage_box({"ticker": p.ticker, "fiscal_year": p.fiscal_year, "item": p.item, "text": p.text})
    with st.form(f"ref_{q['id']}"):
        none = st.checkbox("The filings contain no answer to this question", value=(cur == NO_ANSWER))
        text = st.text_area("Reference answer", value="" if cur == NO_ANSWER else cur, height=110)
        if st.form_submit_button("Save reference answer", type="primary"):
            value = NO_ANSWER if none else text.strip()
            if not value:
                st.warning("Write an answer, or tick the box.")
            else:
                upsert(path, ["question_id", "reference"], ("question_id",), {"question_id": q["id"], "reference": value})
                st.success("Saved.")
                st.rerun()


def step_rate(q: dict, rdir: Path, labels: list[str]) -> None:
    refs = {r["question_id"]: r["reference"] for r in read_rows(rdir / "references.csv")}
    path = rdir / "ratings.csv"
    saved = {r["label"]: r["rating"] for r in read_rows(path) if r["question_id"] == q["id"]}
    st.markdown("**Your reference answer:** " + (esc_md(refs[q["id"]]) if q["id"] in refs else "_not written yet (step 1)_"))
    st.caption("Rate each answer on its own. 2 = correct and complete. 1 = partly correct or incomplete. 0 = wrong, invented, or no answer. "
               "An answer that declines (INSUFFICIENT EVIDENCE) is a 0 unless the filings really have no answer.")
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        a = q["answers"][label]
        with col, st.container(border=True):
            st.markdown(f"#### Answer {label}")
            st.write(esc_md(a["text"]))
            cur = saved.get(label, "")
            choice = st.radio("Your rating", ["2", "1", "0"], index=["2", "1", "0"].index(cur) if cur else None,
                              format_func=SCALE.get, key=f"rate_{q['id']}_{label}")
            if choice is not None and choice != cur:
                upsert(path, ["question_id", "label", "rating"], ("question_id", "label"),
                       {"question_id": q["id"], "label": label, "rating": choice})
                st.rerun()


def step_citations(q: dict, rdir: Path, labels: list[str]) -> None:
    path = rdir / "citations.csv"
    saved = {(r["label"], r["citation"]): r["supports"] for r in read_rows(path) if r["question_id"] == q["id"]}
    st.caption("For each citation: does the cited passage state the main claim of the sentence it is cited for? "
               "If one sentence cites several passages, each passage counts if it supports its own part.")
    shown = False
    for label in labels:
        a = q["answers"][label]
        if not a["cited"]:
            continue
        shown = True
        with st.container(border=True):
            st.markdown(f"#### Answer {label}")
            st.write(esc_md(a["text"]))
            for c in a["cited"]:
                st.markdown(f"**Cited as [{c['ref']}]**")
                passage_box(c)
                cur = saved.get((label, c["ref"]), "")
                choice = st.radio("Does this passage support the claim?", ["1", "0"], index=["1", "0"].index(cur) if cur else None,
                                  format_func={"1": "Yes, it supports it", "0": "No, it does not"}.get, horizontal=True,
                                  key=f"cit_{q['id']}_{label}_{c['ref']}")
                if choice is not None and choice != cur:
                    upsert(path, ["question_id", "label", "citation", "supports"], ("question_id", "label", "citation"),
                           {"question_id": q["id"], "label": label, "citation": c["ref"], "supports": choice})
                    st.rerun()
    if not shown:
        st.success("None of the answers to this question cite a passage, so there is nothing to check here.")


def sidebar_retrieval(name: str) -> None:
    sheet = HUMAN / "judgments" / f"pool_sheet_{name}.csv"
    rows = read_rows(sheet)
    if not rows:
        st.sidebar.caption(f"Step 4 (optional, retrieval judging): you have no sheet yet. Create one safely with "
                           f"`python eval/human_prepare.py --names {name} --sheets-only` (it does not change the answers or labels).")
        return
    done = sum(1 for r in rows if r["rel"].strip())
    st.sidebar.markdown("**Step 4 (optional): retrieval judging**")
    st.sidebar.progress(done / len(rows), text=f"{done} of {len(rows)} passages graded")
    st.sidebar.code('TRADEMESH_JUDGMENTS=eval/human/judgments streamlit run eval/judge_app.py --server.port 8502', language="bash")


# ---------------------------------------------------------------- main

def main() -> None:
    st.set_page_config(page_title="TradeMesh human rating", layout="wide")
    if not (HUMAN / "blind_answers.json").exists():
        st.error("eval/human/blind_answers.json is missing. Run `python eval/human_prepare.py` first.")
        st.stop()
    blind = load_blind()
    questions, labels = blind["questions"], blind["labels"]
    st.title("TradeMesh human rating")
    name = safe_name(st.sidebar.text_input("Your name (letters only)", key="rater"))
    if not name:
        st.info("Type your name in the sidebar to start. Your work is saved under that name.")
        st.stop()
    rdir = HUMAN / "raters" / name
    step = st.sidebar.radio("Step", ["1. Reference answers", "2. Rate answers", "3. Check citations"], key="step")

    refs = {r["question_id"] for r in read_rows(rdir / "references.csv")}
    rated = {}
    for r in read_rows(rdir / "ratings.csv"):
        rated[r["question_id"]] = rated.get(r["question_id"], 0) + 1
    cits = {}
    for r in read_rows(rdir / "citations.csv"):
        cits[r["question_id"]] = cits.get(r["question_id"], 0) + 1
    need_cites = {q["id"]: sum(len(a["cited"]) for a in q["answers"].values()) for q in questions}
    done = {"1. Reference answers": {q["id"] for q in questions if q["id"] in refs},
            "2. Rate answers": {q["id"] for q in questions if rated.get(q["id"], 0) == len(labels)},
            "3. Check citations": {q["id"] for q in questions if cits.get(q["id"], 0) >= need_cites[q["id"]]}}
    for s, d in done.items():
        st.sidebar.progress(len(d) / len(questions), text=f"{s}: {len(d)} of {len(questions)} questions")
    finished = done[step]
    default = next((i for i, q in enumerate(questions) if q["id"] not in finished), 0)
    qid = st.sidebar.selectbox("Question", [q["id"] for q in questions], index=default,
                               format_func=lambda i: f"{'✓' if i in finished else '·'} {i}  {next(x['question'] for x in questions if x['id'] == i)[:46]}")
    sidebar_retrieval(name)
    q = next(x for x in questions if x["id"] == qid)

    st.subheader(f"{q['id']}: {q['question']}")
    if step.startswith("1"):
        step_reference(q, rdir)
    elif step.startswith("2"):
        step_rate(q, rdir, labels)
    else:
        step_citations(q, rdir, labels)
    if all(len(d) == len(questions) for d in done.values()):
        st.balloons()
        st.success("All three steps are complete. Now run `python eval/human_score.py` (see eval/human/README.md).")


main()
