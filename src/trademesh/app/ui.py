"""HTML building blocks for the Streamlit app (cinematic dark-glass look).

Pure functions that return HTML strings, so they are easy to test and the layout code stays short.
Everything dynamic is escaped before it reaches the page (including `$`, so Streamlit never
mistakes two money amounts for a LaTeX formula). Icons come from `icons.py` (Phosphor, bold) and
photographs from `static/` (see `static/credits.json`; shown in the footer).
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from ..types import Passage
from .icons import PATHS

HERE = Path(__file__).parent
STATIC_URL = "/app/static"          # served by Streamlit (server.enableStaticServing, folder app/static)


def esc(s) -> str:
    return html.escape(str(s), quote=True).replace("$", "&#36;")


def css() -> str:
    return "<style>" + (HERE / "cinematic.css").read_text(encoding="utf-8") + "</style>"


def icon(name: str, size: int = 16) -> str:
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 256 256" fill="currentColor" '
            f'aria-hidden="true">{PATHS[name]}</svg>')


def img(name: str) -> str:
    return f"{STATIC_URL}/{name}"


def credits() -> list[dict]:
    try:
        rows = json.loads((HERE / "static" / "credits.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [{"file": f, "title": t, "license": lic, "author": a, "url": u} for f, t, lic, a, u in rows]


# ---------------------------------------------------------------- layout blocks

def hero() -> str:
    """Two-line headline with an inline pill photo, and a floating photo that overlaps from the bottom right."""
    return ('<section class="tm-hero"><h1 class="tm-hero-h1">Ask the filings. '
            f"<span class=\"inl\" style=\"background-image:url('{img('nyse-sign.jpg')}')\"></span> "
            'Trust every <em>number</em>.</h1>'
            '<p class="tm-hero-p">Search 10-K filings, get a cited answer, and see which passage backs each claim.</p>'
            f'<figure class="tm-hero-img"><img src="{img("hero-wall-street.jpg")}" alt="Wall Street seen from a rooftop, 1930s"></figure></section>')


def marquee(companies: list[tuple[str, str]]) -> str:
    """Endless ticker strip of the companies in the corpus, with the product's four promises between them."""
    feats = [("shield", "Verified numbers"), ("search", "Spelling-tolerant search"), ("quotes", "Cited answers"), ("db", "Evidence you can open")]
    items = []
    for i, (ticker, name) in enumerate(companies):
        items.append(f'<span class="it"><span class="tk">{esc(ticker)}</span>{esc(name)}</span>')
        ic, txt = feats[i % len(feats)]
        items.append(f'<span class="it">{icon(ic, 18)}{esc(txt)}</span>')
    if not items:
        return ""
    half = "".join(items) * 2          # long enough that the loop never shows a gap
    return f'<div class="tm-marquee" aria-hidden="true"><div class="track">{half}{half}</div></div>'


def footer() -> str:
    repo = "https://github.com/Deltasthicc/IRFinalProj"
    cr = "".join(f'<li><a href="{esc(c["url"])}" target="_blank" rel="noopener">{esc(c["title"][:58])}</a> '
                 f'({esc(c["license"])}{"" if "nknown" in c["author"] else ", " + esc(c["author"])})</li>' for c in credits())
    return ('<footer class="tm-foot"><h2>Every answer, <em>checked</em> against the filing.</h2><nav>'
            f'<a href="{repo}" target="_blank" rel="noopener">Source on GitHub</a>'
            f'<a href="{repo}#readme" target="_blank" rel="noopener">Read the README</a>'
            '<span>CSD358 Information Retrieval, Monsoon 2026</span></nav>'
            f'<details class="cr"><summary>Image and icon credits</summary><ul>{cr}'
            '<li>Icons: <a href="https://phosphoricons.com" target="_blank" rel="noopener">Phosphor Icons</a> (MIT)</li></ul></details></footer>')


def heading(text: str, ico: str | None = None, small: str = "") -> str:
    return f'<div class="tm-h">{icon(ico, 26) if ico else ""}{esc(text)}<small>{esc(small)}</small></div>'


def _led(state: str) -> str:
    return f'<i class="tm-led {state}"></i>' if state else ""


def chips(items: list[tuple[str, str, str]]) -> str:
    """items = (label, value, led) with led in {"on", "bad", ""}."""
    cells = "".join(f'<div class="tm-chip"><b>{esc(k)}</b><span>{_led(led)}{esc(v)}</span></div>' for k, v, led in items)
    return f'<div class="tm-chips">{cells}</div>'


def hazard(messages: list[str]) -> str:
    if not messages:
        return ""
    return '<div class="tm-hazard">' + icon("warn", 18) + "<b>Caution</b>" + "".join(f"<span>{esc(m)}</span>" for m in messages) + "</div>"


STEP_INFO = {   # one-line explanations, shown when a pipeline slice expands
    "Query": "Your question, as typed.",
    "Entity": "Company names found, with spelling fixed.",
    "Retrieve": "Top passages for the topic, kept to the named company.",
    "Rerank": "A cross-encoder re-scores the best candidates.",
    "Context": "Five labelled passages go to the model.",
    "LLM": "Answers only from the passages, temperature 0, cached.",
    "Verify": "Checks every citation and every number.",
    "Decision": "Shows the answer, or says there is not enough evidence.",
    "Abstain": "Declines when the model says the evidence is insufficient.",
}


def pipeline(steps: list[tuple[str, str, str]]) -> str:
    """steps = (label, value, state) with state in {"ok", "warn", "skip"}. Slices expand on hover."""
    cells = "".join(f'<div class="tm-cell {s}"><i>{esc(k)}</i><b>{esc(v)}</b>'
                    f'{f"<p class=d>{esc(STEP_INFO[k])}</p>" if k in STEP_INFO else ""}</div>'
                    for k, v, s in steps)
    return f'<div class="tm-pipe">{cells}</div>'


def panel(tag: str, body: str, alert: bool = False) -> str:
    return f'<div class="tm-panel{" alert" if alert else ""}"><span class="tm-tag">{esc(tag)}</span>{body}</div>'


def bar(frac: float, tone: str = "", marker: float | None = None) -> str:
    pct = max(0.0, min(1.0, frac)) * 100
    mk = f'<u style="left:{max(0.0, min(1.0, marker)) * 100:.1f}%"></u>' if marker is not None else ""
    return f'<div class="tm-bar {tone}"><i style="width:{pct:.1f}%"></i>{mk}</div>'


def bar_rows(rows: list[tuple[str, float, str]], vmax: float | None = None, tone: str = "") -> str:
    """Labelled horizontal bars: rows = (label, value, value text)."""
    vmax = vmax or max((v for _, v, _ in rows), default=1.0) or 1.0
    return "".join(f'<div class="tm-row"><span>{esc(k)}</span>{bar(v / vmax, tone)}<em>{esc(t)}</em></div>' for k, v, t in rows)


def tiles(items: list[tuple[str, str, str, bool]], bento: bool = False) -> str:
    """items = (label, big value, sub text, pending). bento=True lays them out as a gapless 4x2 bento grid."""
    cells = "".join(f'<div class="tm-tile{" pending" if p else ""}"><b>{esc(v)}</b><span>{esc(k)}</span>'
                    f'<span style="margin-top:.15rem">{esc(sub)}</span></div>' for k, v, sub, p in items)
    return f'<div class="tm-grid{" bento" if bento else ""}">{cells}</div>'


def checks(items: list[tuple[str, bool | None, str]]) -> str:
    """items = (name, ok, detail). ok None = not applicable."""
    out = []
    for name, ok, detail in items:
        state, word = ("bad", "Flag") if ok is False else (("na", "N/A") if ok is None else ("ok", "Pass"))
        ic = icon("warn" if ok is False else "check", 16)
        out.append(f'<div class="tm-check {state}"><b>{ic}{esc(name)}<span class="st">{word}</span></b>'
                   f'{f"<small>{esc(detail)}</small>" if detail else ""}</div>')
    return f'<div class="tm-grid">{"".join(out)}</div>'


def empty_state(big: str, line: str, ico: str, image: str = "nyse-sign.jpg") -> str:
    return (f'<div class="tm-empty" style="--bg-img:url({img(image)})">{icon(ico, 40)}'
            f'<div class="big">{esc(big)}</div><p>{esc(line)}</p></div>')


def table(columns: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>" for r in rows)
    return f'<table class="tm-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------- text helpers

def cite_html(answer: str) -> str:
    """Escape the answer and turn [P3] into a solid citation tag."""
    return re.sub(r"\[P(\d+)\]", r'<span class="tm-cite">P\1</span>', esc(answer)).replace("\n", "<br>")


def mark_html(snippet: str) -> str:
    """kwic() wraps hits in ** **; show them as highlighted marks."""
    return re.sub(r"\*\*(.+?)\*\*", r"<mark>\1</mark>", esc(snippet))


def passage_card(label: str, p: Passage, score: float | None = None, max_score: float = 1.0,
                 cited: bool = False, dim: bool = False, snippet: str | None = None, preview: int = 230,
                 explain: list[tuple[str, float]] | None = None) -> str:
    sc = ""
    if score is not None:
        sc = f'<span>Score <b>{score:.2f}</b></span><span style="width:90px">{bar(score / max_score if max_score else 0)}</span>'
    body = mark_html(snippet) if snippet is not None else esc(p.text[:preview] + ("..." if len(p.text) > preview else ""))
    more = (f'<details><summary>Full text</summary>{esc(p.text)}</details>'
            if snippet is not None or len(p.text) > preview else "")
    why = (f'<details><summary>Why this score</summary>{bar_rows([(t, w, f"{w:.2f}") for t, w in explain])}</details>'
           if explain else "")
    where = "Other sections" if p.item == "X" else f"Item {p.item}"
    cls = "tm-card" + (" cited" if cited else "") + (" dim" if dim else "")
    return (f'<div class="{cls}"><header><b>{esc(label)}</b><span class="tm-badge">{esc(p.ticker)}</span>'
            f'<span>FY{p.fiscal_year}</span><span>{esc(where)}</span><span>{esc(p.kind)}</span>{sc}</header>'
            f'<div class="body">{body}</div>{more}{why}<details><summary>Source</summary>{esc(p.id)}'
            f'{f" / page {p.page}" if p.page else ""}</details></div>')
