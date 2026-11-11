"""TradeMesh Streamlit app: Ask | Search | Filing graph | Pages.

Run:  streamlit run src/trademesh/app/streamlit_app.py
A tab whose back end is not ready shows a "not available yet" message and never crashes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from trademesh.app import style  # noqa: E402
from trademesh.app.backend import (graph_dot, highlight_citations, load_graph, load_results, load_state,  # noqa: E402
                                   load_vision, AppState)
from trademesh.rag.llm import CacheMiss, LiveCallCapReached  # noqa: E402
from trademesh.rag.pipeline import RagConfig, RagResult  # noqa: E402
from trademesh.text import content_terms, kwic  # noqa: E402

st.set_page_config(page_title="TradeMesh", page_icon="📈", layout="wide")


@st.cache_resource
def get_state() -> AppState:
    return load_state()


def passage_card(label: str, p, snippet: str | None = None, expanded: bool = False, score: float | None = None,
                 explain: list[tuple[str, float]] | None = None):
    title = f"{label} {p.ticker} | FY{p.fiscal_year} | Item {p.item} | {p.kind}" + (f" | score {score:.3f}" if score is not None else "")
    with st.expander(title, expanded=expanded):
        st.markdown(snippet if snippet else p.text)
        st.caption(f"passage id: {p.id}" + (f" | page {p.page}" if p.page else ""))
        if explain:
            st.markdown("**Why this score** (each query term's BM25 contribution)")
            import pandas as pd
            st.bar_chart(pd.DataFrame({"contribution": [w for _, w in explain]}, index=[t for t, _ in explain]), height=140)


def trust_panel(res: RagResult, tau: float, state: AppState):
    st.subheader("Trust panel")
    if res.abstained:
        st.warning(f"Abstained: {res.abstain_reason}")
    cols = st.columns(3)
    cols[0].metric("Best retrieval score", f"{res.confidence:.3f}", help="best score / number of content words in the query")
    cols[1].metric("Threshold tau", f"{tau:.3f}")
    cols[2].metric("Answer source", "cache" if res.cached else ("live call" if res.cached is False else "none"))
    v = res.verifier
    if v is None:
        st.info("No verifier result: the model was not called or this is a closed-book answer.")
        return
    if res.abstained and res.model_said_insufficient:
        st.success("The model declined to answer (nothing to verify).")
        return
    (st.success if v.passed else st.error)("Verifier: all checks passed" if v.passed else "Verifier: FLAGGED")
    rows = [("Cited passages exist", not v.invalid_cites, ", ".join(f"[P{i}]" for i in v.cited) or "none cited"),
            ("Answer has a citation", bool(v.cited), ""),
            ("Every sentence cited", not v.uncited_sentences, f"{len(v.uncited_sentences)} uncited" if v.uncited_sentences else "")]
    if v.numbers:
        rows.append(("Numbers found in cited passages", not v.ungrounded,
                     "; ".join(f"{n.raw}: " + (("in " + ",".join(f"[P{i}]" for i in n.grounded_in)) if n.grounded_in else
                               ("only in uncited " + ",".join(f"[P{i}]" for i in n.in_uncited) if n.in_uncited else "not found"))
                               for n in v.numbers)))
    else:
        rows.append(("Numbers found in cited passages", True, "no numbers in the answer"))
    for name, ok, detail in rows:
        st.markdown(f"- {'PASS' if ok else '**FLAG**'} {name}" + (f" ({detail})" if detail else ""))
    if state.llm.is_stub:
        st.caption("Stub LLM in use: this panel verifies a quoted sentence, not a model answer.")


def tab_ask(state: AppState, cfg: RagConfig, retriever: str, use_reranker: bool):
    q = st.text_input("Ask about the filings", placeholder="e.g. How does Tesla describe supplier concentration risk?", key="ask_q")
    c1, c2 = st.columns(2)
    closed = c1.toggle("Closed-book (no passages, same model)", value=False,
                       help="The baseline from the RAG study: the model answers from memory, with no evidence and no verification.")
    force_compare = c2.toggle("Compare mode", value=False, help="One sub-query per company. Switches on by itself when 2+ companies are named.")
    if not q.strip():
        return
    pipe = state.pipeline(retriever, cfg, use_reranker)
    try:
        with st.spinner("Retrieving and answering..."):
            res = pipe.answer(q, closed_book=closed, compare=True if force_compare else None)
    except CacheMiss:
        st.error("This question is not in the answer cache and LLM_OFFLINE=1 is set. Unset it to make a live call.")
        return
    except LiveCallCapReached as e:
        st.error(str(e))
        return
    except Exception as e:  # network, key, quota
        st.error(f"The language model call failed: {e}")
        return

    for typed, name in res.corrections:
        st.info(f"Did you mean **{name}**? Searching for it instead of '{typed}'.")
    st.caption(f"Mode: {res.mode} | model: {res.model or 'n/a'} | {state.llm.usage_line()}")

    if res.mode == "closed_book":
        st.subheader("Answer (closed-book, unverified)")
        st.warning("No evidence was used. Treat this as the baseline, not as a sourced answer.")
        st.markdown(res.answer)
        return

    st.subheader("Answer")
    if res.abstained:
        st.error("INSUFFICIENT EVIDENCE: the system declined to answer.")
        if res.answer and res.answer.strip().upper() != "INSUFFICIENT EVIDENCE":
            with st.expander("What the model wrote before it was withheld"):
                st.markdown(highlight_citations(res.answer))
    else:
        st.markdown(highlight_citations(res.answer))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Evidence")
        cited = set(res.verifier.cited) if res.verifier else set()
        if res.mode == "compare":
            cols = st.columns(len(res.by_company))
            for col, (ticker, labels) in zip(cols, res.by_company.items()):
                with col:
                    st.markdown(f"**{ticker}**")
                    for i in labels:
                        passage_card(f"[P{i}]", res.passages[i - 1], expanded=i in cited, score=res.hits[i - 1].score)
        else:
            for i, p in enumerate(res.passages, 1):
                passage_card(f"[P{i}]", p, expanded=i in cited, score=res.hits[i - 1].score)
        if not res.passages:
            st.write("No passages were retrieved.")
    with right:
        trust_panel(res, cfg.tau, state)


def tab_search(state: AppState, cfg: RagConfig):
    q = st.text_input("Search query", key="search_q", placeholder="e.g. supplier concentration")
    names = list(state.retrievers)
    preset = [n for n in ("bm25", "tfidf", "hybrid") if n in names][:2] or names[:1]
    chosen = st.multiselect("Retrievers (pick up to 3 to compare side by side)", names, default=preset, max_selections=3)
    companies = ["All"] + sorted({p.ticker for p in state.passages})
    ticker = st.selectbox("Company filter", companies)
    k = st.slider("Results", 3, 20, 8)
    if not q.strip() or not chosen:
        return
    mentions = state.matcher.find_mentions(q)
    for typed, name in state.matcher.corrections(mentions):
        st.info(f"Did you mean **{name}**? ('{typed}')")
    by_id = {p.id: p for p in state.passages}
    terms = content_terms(state.matcher.strip_mentions(q, mentions) if mentions else q)
    cols = st.columns(len(chosen))
    for col, name in zip(cols, chosen):
        with col:
            st.markdown(f"**{name}**")
            hits = state.retrievers[name].search(q, k=300)
            if ticker != "All":
                hits = [h for h in hits if by_id[h.passage_id].ticker == ticker]
            if not hits:
                st.write("No results.")
            explainer = getattr(state.retrievers[name], "explain", None)
            for rank, h in enumerate(hits[:k], 1):
                p = by_id[h.passage_id]
                passage_card(f"#{rank}", p, snippet=kwic(p.text, terms), score=h.score,
                             explain=explainer(q, h.passage_id) if explainer else None)


def tab_graph():
    graph, why = load_graph()
    if graph is None:
        st.info(f"Filing graph is {why}")
        return
    st.caption("Edge A -> B: A's filing mentions B. PageRank, hubs and authorities are computed by Role A's code.")
    import pandas as pd
    df = pd.DataFrame(graph["nodes"]).sort_values("pagerank", ascending=False)
    corr = graph.get("correlations") or {}
    if corr:
        c = st.columns(3)
        c[0].metric("PageRank vs in-degree (Spearman)", f"{corr.get('pagerank_vs_indegree_spearman', float('nan')):.2f}",
                    help="Near 1 means PageRank adds little beyond counting mentions (H10).")
        c[1].metric("Authority vs in-degree (Spearman)", f"{corr.get('authority_vs_indegree_spearman', float('nan')):.2f}")
        c[2].metric("Edges", graph.get("params", {}).get("n_edges", len(graph["edges"])))
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.graphviz_chart(graph_dot(graph))


def tab_pages():
    vision, why = load_vision()
    if vision is None:
        st.info(f"Page-image retrieval is {why} (the vision track is gated).")
        return
    for q in vision["queries"]:
        st.markdown(f"**{q['query']}**")
        cols = st.columns(max(1, min(4, len(q["results"]))))
        for col, r in zip(cols, q["results"]):
            with col:
                img = Path(r.get("image", ""))
                if img.exists():
                    st.image(str(img), caption=f"{r['ticker']} p.{r['page']} score {r['score']:.2f}")
                else:
                    st.write(f"{r['ticker']} p.{r['page']} (image missing)")


def tab_eval():
    res, why = load_results()
    if res is None:
        st.info(f"Retrieval results are {why}")
        return
    import pandas as pd
    st.caption(f"{'Locked TEST run' if res['split'] == 'test' else 'DEVELOPMENT run (tuning data, not the final result)'}: "
               f"{res['n_needs']} information needs, pooled blind judgments, baseline {res['baseline']}.")
    cols = ["P@5", "P@10", "AP", "MRR", "nDCG@10", "bpref", "judged@10"]
    df = pd.DataFrame(res["overall"]).T[[c for c in cols if c in next(iter(res["overall"].values()))]]
    df = df.rename(columns={"AP": "MAP"})
    st.dataframe(df.style.format("{:.3f}").highlight_max(axis=0, color="#D7ECFF"), use_container_width=True)
    st.markdown("**11-point interpolated precision-recall**")
    st.line_chart(pd.DataFrame(res["pr_11pt"], index=[f"{i / 10:.1f}" for i in range(11)]), height=260)
    st.markdown("**By need type (nDCG@10)**")
    st.dataframe(pd.DataFrame({s: {t: v["nDCG@10"] for t, v in d.items()} for s, d in res["by_type"].items()}).style.format("{:.3f}"),
                 use_container_width=True)
    if res.get("hypotheses"):
        st.markdown("**Hypotheses, decided by rules fixed before the test run**")
        h = pd.DataFrame(res["hypotheses"])
        keep = [c for c in ("id", "comparison", "mean_diff", "ci_low", "ci_high", "wins", "losses", "ties", "verdict") if c in h]
        st.dataframe(h[keep], use_container_width=True, hide_index=True)


def main():
    st.markdown(style.CSS, unsafe_allow_html=True)
    state = get_state()
    st.markdown(style.hero(len(state.passages), len(state.companies), state.is_fixture, len(state.retrievers)), unsafe_allow_html=True)
    for w in state.warnings:
        st.warning(w)

    with st.sidebar:
        st.header("Settings")
        names = list(state.retrievers)
        default = next((n for n in ("hybrid", "bm25") if n in names), names[0])      # the strongest available, not alphabetical luck
        retriever = st.selectbox("Retriever for answers", names, index=names.index(default))
        use_reranker = st.checkbox("Use cross-encoder reranker", value=state.reranker is not None,
                                   disabled=state.reranker is None,
                                   help="Available when Role B's reranker is installed.")
        tau = st.number_input("Abstention threshold tau", min_value=0.0, value=0.0, step=0.1,
                              help="Abstain when the best score is below tau. Tuned on dev queries; 0 = off.")
        lam = st.number_input("Section prior lambda", min_value=0.0, max_value=2.0, value=0.0, step=0.05,
                              help="final = base + lambda * g(Item). 0 = off.")
        st.caption(f"LLM: {state.llm.provider} / {state.llm.model}")
        st.caption(f"Passages: {len(state.passages)} | Companies: {len(state.companies)}")
    cfg = RagConfig(tau=tau, lam=lam)

    ask, search, graph, pages, evaluation = st.tabs(["Ask", "Search", "Filing graph", "Pages", "Evaluation"])
    for tab, fn in ((ask, lambda: tab_ask(state, cfg, retriever, use_reranker)),
                    (search, lambda: tab_search(state, cfg)), (graph, tab_graph), (pages, tab_pages),
                    (evaluation, tab_eval)):
        with tab:
            try:
                fn()
            except Exception as e:  # a broken tab must not take the app down
                st.error(f"This tab hit an error: {e}")


main()
