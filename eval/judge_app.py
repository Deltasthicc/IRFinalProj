"""Blind relevance-judging app for the pooled collection. Run:  streamlit run eval/judge_app.py

Pick your name; the app loads data/judgments/pool_sheet_<name>.csv (from eval/make_pools.py), shows one
information need at a time with its shuffled passages (no system names, no ranks), and saves every
grade straight back to that CSV, so you can stop and resume. Grades: 2 = answers the need,
1 = partly relevant, 0 = not relevant. Do not open pool_key.json while judging.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import streamlit as st

JUDG = Path(os.environ.get("TRADEMESH_JUDGMENTS", Path(__file__).resolve().parents[1] / "data" / "judgments"))
GUIDE = {"2": "2  answers the need", "1": "1  partly relevant", "0": "0  not relevant"}


def load(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def save(path: Path, fields, rows) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


st.set_page_config(page_title="TradeMesh judging", layout="wide")
sheets = sorted(JUDG.glob("pool_sheet_*.csv"))
if not sheets:
    st.error("No pool sheets found. Run `python eval/make_pools.py` first.")
    st.stop()
name = st.sidebar.selectbox("Judge", [s.stem.removeprefix("pool_sheet_") for s in sheets])
path = JUDG / f"pool_sheet_{name}.csv"
fields, rows = load(path)
needs = list(dict.fromkeys(r["query_id"] for r in rows))
done = {q: all(r["rel"].strip() for r in rows if r["query_id"] == q) for q in needs}
st.sidebar.progress(sum(done.values()) / len(needs), text=f"{sum(done.values())} of {len(needs)} needs finished")
st.sidebar.markdown("**2** answers the need  \n**1** partly relevant  \n**0** not relevant")
default = next((i for i, q in enumerate(needs) if not done[q]), 0)
qid = st.sidebar.selectbox("Need", needs, index=default, format_func=lambda q: f"{'✓' if done[q] else '·'} {q}")

items = [r for r in rows if r["query_id"] == qid]
st.header(f"{qid}: {items[0]['need']}")
st.caption(f"{len(items)} passages, shown in random order. Judge each passage on its own, as an answer to this need.")
changed = False
for r in items:
    with st.container(border=True):
        st.markdown(f"**{r['ticker']}**  ·  FY{r['fiscal_year']}  ·  Item {r['item']}  ·  {r['kind']}")
        st.write(r["passage_text"])
        cur = r["rel"].strip()
        choice = st.radio("Grade", ["2", "1", "0"], index=["2", "1", "0"].index(cur) if cur else None,
                          format_func=GUIDE.get, key=f"{name}|{qid}|{r['passage_id']}", horizontal=True)
        if choice is not None and choice != cur:
            r["rel"] = choice
            changed = True
if changed:
    save(path, fields, rows)
    st.rerun()
