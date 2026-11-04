"""Small text helpers shared by C's modules (tokenising, stop words, KWIC snippets)."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the their
    this to was were what which who with does do did about into than then there these those
    over under per vs versus between across compare describe say says said company companies""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens. Decimal numbers such as 1.2 stay one token."""
    return _TOKEN.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """Tokens without stop words, used for query-length normalisation and highlighting."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def kwic(text: str, terms: list[str], width: int = 80) -> str:
    """Key-word-in-context snippet: a window around the first query term found in `text`.

    Matched terms are wrapped in ** ** so Streamlit markdown shows them bold.
    """
    terms = [t for t in dict.fromkeys(terms) if t]
    if not terms:
        return text[: 2 * width] + ("..." if len(text) > 2 * width else "")
    pattern = re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), re.I)
    m = pattern.search(text)
    if m is None:
        start, end = 0, min(len(text), 2 * width)
    else:
        start, end = max(0, m.start() - width), min(len(text), m.end() + width)
    snippet = pattern.sub(lambda mm: f"**{mm.group(0)}**", text[start:end])
    return ("..." if start > 0 else "") + snippet.strip() + ("..." if end < len(text) else "")
