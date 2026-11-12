"""TradeMesh Streamlit app: Ask | Search | Graph | Pages | Evaluation | System.

Run:  streamlit run src/trademesh/app/streamlit_app.py
Look: dark glass, "cinematic" (see ui.py and cinematic.css). A tab whose back end is not ready shows a
"no signal" panel and never crashes.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from trademesh.app import ui  # noqa: E402
from trademesh.app.backend import (EXAMPLES, AppState, graph_dot, headline_numbers, load_entity_eval, load_graph,  # noqa: E402
                                   load_rag_study, load_results, load_state, load_vision, readiness)
from trademesh.rag.llm import CacheMiss, LiveCallCapReached  # noqa: E402
from trademesh.rag.pipeline import RagConfig, RagResult  # noqa: E402
from trademesh.text import content_terms, kwic  # noqa: E402

st.set_page_config(page_title="TradeMesh", page_icon="■", layout="wide")

@st.cache_resource(show_spinner="Building the search indexes for the filings (first load only, about 15 seconds)")
def get_state() -> AppState:
    return load_state()


def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------- header

def status_chips(state: AppState) -> str:
    live = not state.llm.is_stub
    return ui.chips([
        ("Corpus", "Test fixture" if state.is_fixture else "10-K filings", "bad" if state.is_fixture else "on"),
        ("Retrievers", "Dev stub" if state.using_stub else f"{len(state.retrievers)} live", "bad" if state.using_stub else "on"),
        ("LLM", f"{state.llm.provider} / {state.llm.model}" if live else "Stub", "on" if live else "bad"),
        ("Passages", f"{len(state.passages):,}", ""),
        ("Companies", str(len(state.companies)), ""),
        ("LLM calls", f"{state.llm.live_calls} live / {state.llm.cache_hits} cache", ""),
    ])


def header(state: AppState):
    """Draw the hero, call-to-action buttons, marquee and status; return the chip slot so call counts refresh after a run."""
    md(ui.css() + ui.hero())
    c1, c2, _ = st.columns([1.1, 1.5, 3])
    c1.button("Ask with a typo", type="primary", key="cta_typo", width="stretch",
              on_click=lambda: st.session_state.update(ask_q=EXAMPLES[0][1]))
    c2.button("Compare two companies", key="cta_compare", width="stretch",
              on_click=lambda: st.session_state.update(ask_q=EXAMPLES[1][1]))
    md(ui.marquee([(c["ticker"], c["name"]) for c in state.companies]))
    slot = st.empty()
    slot.markdown(status_chips(state), unsafe_allow_html=True)
    md(ui.hazard(state.warnings))
    return slot


# ---------------------------------------------------------------- ask

def ask_pipeline(res: RagResult) -> str:
    v = res.verifier
    tickers = ", ".join(dict.fromkeys(m.ticker for m in res.mentions)) or "none"
    llm_called = res.cached is not None
    verdict = ("skip", "-") if v is None else (("ok", "Pass") if v.passed else ("warn", "Flag"))
    return ui.pipeline([
        ("Query", f"{len(res.question.split())} words", "ok"),
        ("Entity", tickers + (f" / {len(res.corrections)} fixed" if res.corrections else ""), "ok" if res.mentions else "skip"),
        ("Retrieve", f"{len(res.passages)} passages", "ok" if res.passages else "warn"),
        ("Context", f"P1-P{len(res.passages)}" if res.passages else "-", "ok" if res.passages else "skip"),
        ("LLM", ("Cache" if res.cached else "Live") if llm_called else "Skipped", "ok" if llm_called else "skip"),
        ("Verify", verdict[1], verdict[0]),
        ("Decision", "Abstain" if res.abstained else "Answer", "warn" if res.abstained else "ok"),
    ])


def trust_panel(res: RagResult, tau: float, state: AppState) -> None:
    md('<span class="tm-pin"></span>' + ui.heading("Trust panel", "shield", "verifier"))
    scale = max(res.confidence, tau, 1e-9) * 1.3
    md(ui.panel("Retrieval score (for inspection)",
                ui.bar(res.confidence / scale, marker=(tau / scale) if tau > 0 else None) +
                f'<span style="font-size:.78rem">score <b>{res.confidence:.2f}</b> / tau <b>{"off" if tau == 0 else f"{tau:.2f}"}</b></span>'
                '<br><span style="font-size:.72rem;opacity:.65">Not used to decide abstention: in our study this score did not separate '
                'answerable from unanswerable questions. The system abstains when the model itself says the evidence is insufficient.</span>'))
    if res.abstained:
        md(ui.panel("Abstained", esc_reason(res), alert=True))
    v = res.verifier
    if v is None:
        md(ui.checks([("Verifier", None, "no model answer to check (closed-book, or abstained before the call)")]))
        return
    if res.abstained and res.model_said_insufficient:
        md(ui.checks([("Model declined", True, "nothing to verify")]))
        return
    md(ui.checks([
        ("Citations exist", not v.invalid_cites, ", ".join(f"P{i}" for i in v.cited) or "none cited"),
        ("Answer cited", bool(v.cited), ""),
        ("Every sentence cited", not v.uncited_sentences, f"{len(v.uncited_sentences)} without citation" if v.uncited_sentences else ""),
        ("Numbers grounded", (not v.ungrounded) if v.numbers else None, "" if v.numbers else "no numbers in answer"),
    ]))
    if v.numbers:
        chips = "".join(
            f'<span class="tm-num{"" if n.grounded_in else " bad"}">{ui.esc(n.raw)} '
            f'{"in " + "/".join(f"P{i}" for i in n.grounded_in) if n.grounded_in else ("only in uncited P" + "/".join(map(str, n.in_uncited)) if n.in_uncited else "not found")}</span>'
            for n in v.numbers)
        md(ui.panel("Numbers in the answer", chips))
    md(ui.panel("Run", f'Model {ui.esc(res.model or "n/a")}<br>Source {"cache" if res.cached else "live call"}<br>Mode {ui.esc(res.mode)}'
                + ('<br><span style="opacity:.65">Stub LLM: this verifies a quoted sentence, not a model answer.</span>' if state.llm.is_stub else "")))


def esc_reason(res: RagResult) -> str:
    return f'<span style="font-size:.9rem">{ui.esc(res.abstain_reason)}</span>'


def evidence(res: RagResult) -> None:
    md(ui.heading("Evidence", "db", f"{len(res.passages)} passages"))
    if not res.passages:
        md(ui.panel("No evidence", "No passage was retrieved."))
        return
    cited = set(res.verifier.cited) if res.verifier else set()
    top = max((h.score for h in res.hits), default=1.0) or 1.0

    def card(i: int) -> str:
        return ui.passage_card(f"P{i}", res.passages[i - 1], res.hits[i - 1].score, top,
                               cited=i in cited, dim=bool(cited) and i not in cited)

    if res.mode == "compare":
        cols = st.columns(len(res.by_company))
        for col, (ticker, labels) in zip(cols, res.by_company.items()):
            with col:
                md(f'<span class="tm-tag">{ui.esc(ticker)}</span>' + "".join(card(i) for i in labels))
    else:
        md("".join(card(i) for i in range(1, len(res.passages) + 1)))


def corrected(typed: str, name: str) -> str:
    return (f'<div class="tm-panel alert">{ui.icon("warn", 16)} Corrected &nbsp;<b>{ui.esc(typed)}</b> '
            f'&gt;&gt;&gt; <b>{ui.esc(name)}</b></div>')


def tab_ask(state: AppState, cfg: RagConfig, retriever: str, use_reranker: bool) -> None:
    md(ui.heading("Ask the filings", "chat", "cited and verified"))
    cols = st.columns(len(EXAMPLES))
    for col, (label, q) in zip(cols, EXAMPLES):
        col.button(label, key=f"ex_{label}", width="stretch",
                   on_click=lambda q=q: st.session_state.update(ask_q=q))
    q = st.text_input("Question", placeholder="e.g. How does Tesla describe supplier concentration risk?", key="ask_q")
    c1, c2, _ = st.columns([2, 2, 3])
    closed = c1.toggle("Closed-book", help="Same model, no passages: the no-evidence baseline of the RAG study.")
    force_compare = c2.toggle("Compare", help="One sub-query per company. Switches on by itself when 2+ companies are named.")
    if not q.strip():
        md(ui.empty_state("Awaiting query", "Pick an example above or type a question", "search"))
        return
    pipe = state.pipeline(retriever, cfg, use_reranker)
    try:
        with st.spinner("Retrieving, answering, verifying"):
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
        md(corrected(typed, name))
    if res.mode == "closed_book":
        md(ui.panel("Closed-book / no evidence / unverified", f'<div class="tm-answer">{ui.cite_html(res.answer)}</div>', alert=True))
        return
    md(ask_pipeline(res))
    left, right = st.columns([3, 2])
    with left:
        md(ui.heading("Answer", "chat"))
        if res.abstained:
            md(ui.panel("Abstained", '<div class="tm-answer"><b>Insufficient evidence.</b> The system declined to answer.</div>', alert=True))
            if res.answer and res.answer.strip().upper() != "INSUFFICIENT EVIDENCE":
                md(ui.panel("What the model wrote before it was withheld", f'<div class="tm-answer" style="font-size:1rem">{ui.cite_html(res.answer)}</div>'))
        else:
            md(ui.panel("Answer", f'<div class="tm-answer">{ui.cite_html(res.answer)}</div>'))
        evidence(res)
    with right:
        trust_panel(res, cfg.tau, state)


# ---------------------------------------------------------------- search

def tab_search(state: AppState) -> None:
    md(ui.heading("Search engine", "search", "side by side"))
    q = st.text_input("Query", key="search_q", placeholder="e.g. supplier concentration")
    names = list(state.retrievers)
    preset = [n for n in ("bm25", "tfidf", "hybrid") if n in names][:2] or names[:1]
    c1, c2, c3 = st.columns([3, 2, 1])
    chosen = c1.multiselect("Retrievers (up to 3)", names, default=preset, max_selections=3)
    ticker = c2.selectbox("Company", ["All"] + sorted({p.ticker for p in state.passages}))
    k = c3.number_input("Top K", 3, 20, 8)
    if not q.strip() or not chosen:
        md(ui.empty_state("No query", "Enter terms to rank passages", "search"))
        return
    mentions = state.matcher.find_mentions(q)
    for typed, name in state.matcher.corrections(mentions):
        md(corrected(typed, name))
    by_id = {p.id: p for p in state.passages}
    terms = content_terms(state.matcher.strip_mentions(q, mentions) if mentions else q)
    for col, name in zip(st.columns(len(chosen)), chosen):
        with col:
            retr = state.retrievers[name]
            hits = retr.search(q, k=300)
            if ticker != "All":
                hits = [h for h in hits if by_id[h.passage_id].ticker == ticker]
            hits = hits[: int(k)]
            md(ui.heading(name.split(" (")[0], "bars", f"{len(hits)} hits"))
            if not hits:
                md(ui.panel("Result", "No results."))
                continue
            by_item = Counter(by_id[h.passage_id].item for h in hits)
            md(ui.panel("Hits by 10-K item", ui.bar_rows([("Other" if it == "X" else f"Item {it}", n, str(n)) for it, n in by_item.most_common(6)])))
            top = hits[0].score or 1.0
            explainer = getattr(retr, "explain", None)
            md("".join(ui.passage_card(f"#{r}", by_id[h.passage_id], h.score, top, snippet=kwic(by_id[h.passage_id].text, terms),
                                       explain=explainer(q, h.passage_id) if explainer else None)
                       for r, h in enumerate(hits, 1)))


# ---------------------------------------------------------------- graph / pages

def tab_graph() -> None:
    md(ui.heading("Filing graph", "graph", "who mentions whom"))
    graph, why = load_graph()
    if graph is None:
        md(ui.empty_state("No signal", f"Filing graph is {why}", "graph", "manhattan-1931.jpg"))
        return
    nodes = sorted(graph["nodes"], key=lambda n: -n["pagerank"])
    corr = graph.get("correlations") or {}
    tiles = [("Companies", str(len(nodes)), "nodes", False),
             ("Mentions", str(graph.get("params", {}).get("n_edges", len(graph["edges"]))), "edges", False),
             ("Top PageRank", nodes[0]["ticker"] if nodes else "-", "highest score", False)]
    if corr:
        tiles.append(("PageRank vs in-degree", f'{corr.get("pagerank_vs_indegree_spearman", float("nan")):.2f}',
                      "Spearman. Near 1 means PageRank adds little beyond counting mentions (H10)", False))
    md(ui.tiles(tiles))
    md(ui.panel("PageRank (top 12)", ui.bar_rows([(n["ticker"], n["pagerank"], f'{n["pagerank"]:.3f}') for n in nodes[:12]])))
    st.graphviz_chart(graph_dot(graph))
    cols = [c for c in ("ticker", "pagerank", "hub", "authority", "in_degree") if c in nodes[0]]
    md(ui.table([c.replace("_", " ") for c in cols],
                [[round(n[c], 4) if isinstance(n[c], float) else n[c] for c in cols] for n in nodes]))


def tab_pages() -> None:
    md(ui.heading("Page retrieval", "page", "vision track, gated"))
    vision, why = load_vision()
    if vision is None:
        md(ui.empty_state("No signal", f"Page-image retrieval is {why} (the vision track is gated)", "page", "manhattan-1931.jpg"))
        return
    for q in vision["queries"]:
        md(f'<span class="tm-tag">{ui.esc(q["query"])}</span>')
        cols = st.columns(max(1, min(4, len(q["results"]))))
        for col, r in zip(cols, q["results"]):
            with col:
                img = Path(r.get("image", ""))
                if img.exists():
                    st.image(str(img), caption=f"{r['ticker']} p.{r['page']} / {r['score']:.2f}")
                else:
                    md(ui.panel(f"{r['ticker']} p.{r['page']}", "Image missing"))


# ---------------------------------------------------------------- evaluation

CONFIG_NAMES = {"closed_book": "Closed-book", "bm25_rag": "BM25-RAG", "best_rag": "Best-RAG (hybrid + rerank)"}


def eval_retrieval() -> None:
    res, why = load_results()
    if res is None:
        md(ui.empty_state("No signal", f"Retrieval results are {why}", "bars", "manhattan-1931.jpg"))
        return
    import pandas as pd
    md(ui.heading("Retrieval", "bars", "locked test run" if res["split"] == "test" else "development run"))
    st.caption(f"{'Locked TEST run' if res['split'] == 'test' else 'DEVELOPMENT run (tuning data, not the final result)'}: "
               f"{res['n_needs']} information needs, pooled judgments (LLM-assisted, no human validation), baseline {res['baseline']}.")
    ndcg = sorted(((k, v["nDCG@10"]) for k, v in res["overall"].items() if "nDCG@10" in v), key=lambda t: -t[1])
    md(ui.panel("nDCG@10 by system", ui.bar_rows([(k, v, f"{v:.3f}") for k, v in ndcg], vmax=1.0)))
    cols = ["P@5", "P@10", "AP", "MRR", "nDCG@10", "bpref", "judged@10"]
    df = pd.DataFrame(res["overall"]).T[[c for c in cols if c in next(iter(res["overall"].values()))]].rename(columns={"AP": "MAP"})
    st.dataframe(df.style.format("{:.3f}"), width="stretch")
    md(ui.heading("Precision-recall", "trend", "11-point interpolated"))
    st.line_chart(pd.DataFrame(res["pr_11pt"], index=[f"{i / 10:.1f}" for i in range(11)]), height=260)
    st.markdown("**By need type (nDCG@10)**")
    st.dataframe(pd.DataFrame({s: {t: v["nDCG@10"] for t, v in d.items()} for s, d in res["by_type"].items()}).style.format("{:.3f}"),
                 width="stretch")
    if res.get("hypotheses"):
        st.markdown("**Hypotheses, decided by rules fixed before the test run**")
        h = pd.DataFrame(res["hypotheses"])
        keep = [c for c in ("id", "comparison", "mean_diff", "ci_low", "ci_high", "wins", "losses", "ties", "verdict") if c in h]
        st.dataframe(h[keep], width="stretch", hide_index=True)


def eval_rag() -> None:
    study, why = load_rag_study()
    if study is None:
        md(ui.empty_state("No signal", f"RAG study results are {why}", "flask", "manhattan-1931.jpg"))
        return
    import pandas as pd
    md(ui.heading("RAG answers", "flask", "20 test questions"))
    md(ui.panel("Read this first", "All answer ratings and citation checks are LLM-assisted (two passes by one language model); no human validated them. "
                "20 test questions, one small generator (gemini-3.5-flash-lite).", alert=True))
    pct = lambda x: "n/a" if x is None else f"{x:.3f}"
    cfgs = study["scores"]["configs"]
    md(ui.panel("Mean answer rating (0 to 2)", ui.bar_rows([(CONFIG_NAMES[k], cfgs[k]["mean_rating"], f'{cfgs[k]["mean_rating"]:.2f}')
                                                           for k in CONFIG_NAMES if k in cfgs], vmax=2.0)))
    rows = []
    for k, label in CONFIG_NAMES.items():
        c = cfgs.get(k)
        if c:
            rows.append({"Configuration": label, "Mean rating (0-2)": f"{c['mean_rating']:.3f}", "Rated 2": f"{c['share_rated_2']:.0%}",
                         "Citation precision": pct(c.get("citation_precision")), "Number grounding": pct(c.get("number_grounding")),
                         "Verifier pass": pct(c.get("verifier_pass_rate"))})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if "kappa" in study["scores"]:
        st.caption(f"Cohen's kappa between the two rating passes: {study['scores']['kappa']:.3f} on {study['scores']['n_double_rated']} answers "
                   f"(same model both times, so probably optimistic).")
    b = study.get("bootstrap")
    if b:
        st.markdown(f"**Paired bootstrap over questions** ({b['resamples']:,} resamples; difference in mean rating, 95% interval)")
        pairs = []
        for k, v in b["pairs"].items():
            a, c = k.split(" - ")
            pairs.append({"Comparison": f"{CONFIG_NAMES.get(a, a)} minus {CONFIG_NAMES.get(c, c)}", "Difference": f"{v['mean_diff']:+.3f}",
                          "95% interval": f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]",
                          "Better / worse / tied": f"{v['better']} / {v['worse']} / {v['tied']}",
                          "Excludes 0": "yes" if v["ci95"][0] > 0 or v["ci95"][1] < 0 else "no"})
        st.dataframe(pd.DataFrame(pairs), width="stretch", hide_index=True)
        bt = pd.DataFrame(b["by_type"]).T.rename(columns=CONFIG_NAMES)
        n = bt.pop("n")
        bt.insert(0, "n", n.astype(int))
        st.markdown("**By question type** (6 to 7 questions each: descriptive only)")
        st.dataframe(bt.style.format({c: "{:.2f}" for c in bt.columns if c != "n"}), width="stretch")


def eval_abstention() -> None:
    study, why = load_rag_study()
    if study is None or not study.get("abstention"):
        md(ui.empty_state("No signal", f"Abstention results are {why or 'not available yet: run eval/abstention_eval.py.'}", "scales", "manhattan-1931.jpg"))
        return
    import pandas as pd
    md(ui.heading("Abstention", "scales", "should the system decline?"))

    def rows(label, d):
        out = []
        for part, name in (("dev_tuned", "Dev (tuned on)"), ("test", "Test (headline)")):
            m = d[part]
            out.append({"Rule": label, "Items": name, "Precision": f"{m['precision']:.3f}", "Recall": f"{m['recall']:.3f}",
                        "F1": f"{m['f1']:.3f}", "tp / fp / fn / tn": f"{m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}"})
        return out

    table = rows("Model verdict (tau tuned on dev = 0, so score rule off)", study["abstention"])
    if study.get("abstention_score_only"):
        table = rows(f"Retrieval score only (tau = {study['abstention_score_only']['tau']:.4f})", study["abstention_score_only"]) + table
    st.markdown("**Should the system abstain? Positive class = should abstain** (test: 20 answerable and 6 unanswerable questions)")
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    st.caption("The retrieval score comes from rank fusion, so it reflects ranks, not match quality, and does not separate the two groups. "
               "6 of 6 correct abstentions has a 95% lower bound of about 0.54, and the unanswerable questions are easy (clearly outside the corpus). "
               "The score-only numbers were seen on the test items before the model-verdict rule was chosen.")


def tab_eval() -> None:
    retrieval, rag, abstention = st.tabs(["Retrieval", "RAG answers", "Abstention"])
    for tab, fn in ((retrieval, eval_retrieval), (rag, eval_rag), (abstention, eval_abstention)):
        with tab:
            fn()


# ---------------------------------------------------------------- system

def tab_system(state: AppState) -> None:
    md(ui.heading("System", "cpu", "status, results, tools"))
    md(ui.pipeline([
        ("Query", "Entities + spelling", "ok"), ("Retrieve", "BM25 / dense / hybrid", "ok"),
        ("Rerank", "Cross-encoder", "ok" if state.reranker else "skip"), ("Context", "Top-5 [P1..P5]", "ok"),
        ("LLM", "Temp 0 / cached", "ok"), ("Verify", "Cites + numbers", "ok"), ("Abstain", "Model verdict", "ok"),
    ]))

    md(ui.heading("Readiness", "check", "what is wired up"))
    md(ui.checks([(name, ok, detail) for name, ok, detail in readiness(state)]))

    md(ui.heading("Results", "bars", "real numbers only"))
    md(ui.tiles(headline_numbers(), bento=True))
    ev = load_entity_eval()
    if ev:
        md(ui.panel("Entity accuracy by typo kind",
                    ui.bar_rows([(k.capitalize(), v["correct"] / v["n"], f'{v["correct"]}/{v["n"]}') for k, v in ev["by_kind"].items()], vmax=1.0)))
        if ev["failures"]:
            md(ui.panel("Missed", " ".join(f'<span class="tm-num bad">{ui.esc(f["typo"])} &gt; {ui.esc(f["expected"])}</span>' for f in ev["failures"]), alert=True))

    md(ui.heading("Typo lab", "target", "try the spelling module"))
    t = st.text_input("Type a company name or ticker, misspelt on purpose", key="typo_q", placeholder="e.g. Micorsoft")
    if t.strip():
        m = state.matcher.match_string(t)
        if m:
            md(ui.panel("Match", f'<span class="tm-answer"><b>{ui.esc(t)}</b> &gt;&gt;&gt; <b>{ui.esc(state.matcher.names[m.ticker])}</b> ({ui.esc(m.ticker)})</span>'
                                 f'<br><span style="font-size:.78rem;opacity:.7">Edits {m.distance} / Jaccard {m.jaccard:.2f} / via {ui.esc(m.surface)}</span>'))
        else:
            md(ui.panel("No match", f"Nothing within 2 edits (1 for short strings). The dictionary has {len(state.companies)} companies.", alert=True))


# ---------------------------------------------------------------- main

def main() -> None:
    state = get_state()
    with st.sidebar:
        md('<span class="tm-tag">Controls</span>')
        names = list(state.retrievers)
        default = next((n for n in ("hybrid", "bm25") if n in names), names[0])      # the strongest available, not alphabetical luck
        retriever = st.selectbox("Retriever for answers", names, index=names.index(default))
        use_reranker = st.checkbox("Cross-encoder reranker", value=state.reranker is not None,
                                   disabled=state.reranker is None, help="Available when Role B's reranker is installed.")
        tau = st.number_input("Abstention threshold tau", min_value=0.0, value=0.0, step=0.1,
                              help="Abstain when the best score is below tau. Tuned on the dev queries it came out as 0 (off): the model's own verdict did the work.")
        lam = st.number_input("Section prior lambda", min_value=0.0, max_value=2.0, value=0.0, step=0.05,
                              help="final = base + lambda * g(Item). 0 = off.")
    cfg = RagConfig(tau=tau, lam=lam)
    chip_slot = header(state)

    tabs = st.tabs(["Ask", "Search", "Graph", "Pages", "Evaluation", "System"])
    runners = [lambda: tab_ask(state, cfg, retriever, use_reranker), lambda: tab_search(state), tab_graph, tab_pages,
               tab_eval, lambda: tab_system(state)]
    for tab, fn in zip(tabs, runners):
        with tab:
            try:
                fn()
            except Exception as e:  # a broken tab must not take the app down
                st.error(f"This tab hit an error: {e}")
    chip_slot.markdown(status_chips(state), unsafe_allow_html=True)   # refresh LLM call counts
    md(ui.footer())


main()
