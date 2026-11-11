"""Look and feel: one small CSS block and a header, kept apart from the app logic."""
from __future__ import annotations

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.tm-hero {background: linear-gradient(120deg, #0B2A6F 0%, #0B5FFF 60%, #3AA0FF 100%); color: #fff;
          border-radius: 14px; padding: 1.1rem 1.5rem; margin-bottom: 1rem;}
.tm-hero h1 {color: #fff; margin: 0; font-size: 2rem; letter-spacing: -0.5px;}
.tm-hero p {margin: .2rem 0 0 0; opacity: .92; font-size: .98rem;}
.tm-pill {display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px; padding: .05rem .7rem;
          margin: .5rem .35rem 0 0; font-size: .8rem;}
div[data-testid="stExpander"] {border-radius: 10px; border: 1px solid #DDE5F2;}
div[data-testid="stMetric"] {background: #F2F5FA; border-radius: 10px; padding: .6rem .9rem;}
button[data-baseweb="tab"] {font-weight: 600;}
</style>
"""


def hero(n_passages: int, n_companies: int, fixture: bool, n_retrievers: int = 0) -> str:
    badge = ('<span class="tm-pill" style="background:#FFB020;color:#3b2a00">TEST FIXTURE: invented text</span>' if fixture else
             f'<span class="tm-pill">{n_passages:,} passages</span><span class="tm-pill">{n_companies} companies</span>')
    return (f'<div class="tm-hero"><h1>TradeMesh</h1>'
            f'<p>Search and cited question answering over SEC 10-K filings. Every claim carries a citation, '
            f'a verifier checks the numbers, and the system says so when the evidence is thin.</p>'
            f'{badge}<span class="tm-pill">{n_retrievers} retrievers to compare</span></div>')
