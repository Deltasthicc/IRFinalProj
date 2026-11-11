"""Blocks of one Item -> Passages (Role A).

Text: consecutive paragraphs are packed up to `max_words`; an over-long paragraph is split at sentence
ends. A short heading-like line is carried into the next passage so a passage keeps its context.
Tables: ONE passage per table (split only when very large), written as
    <caption>
    Columns: <header cells>
    <row label> | <value> | <value> ...
so every figure stays attached to its row label and column header (proposal 9.6).
"""
from __future__ import annotations

import re

from ..types import Passage
from .parse import Block

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def words(s: str) -> int:
    return len(s.split())


def is_heading(text: str) -> bool:
    w = words(text)
    return 0 < w <= 12 and not text.rstrip().endswith((".", ";", ",")) and text[:1].isupper()


def split_long(par: str, max_words: int) -> list[str]:
    out, cur = [], []
    for s in SENT.split(par):
        if cur and words(" ".join(cur)) + words(s) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def table_passages(rows: list[list[str]], caption: str, max_words: int) -> list[str]:
    header, body = rows[0], rows[1:]
    lead = [caption] if caption else []
    lead.append("Columns: " + " | ".join(header))
    texts, cur = [], []
    for r in body:
        line = " | ".join(r)
        if cur and words(" ".join(lead + cur)) + words(line) > max_words:
            texts.append("\n".join(lead + cur)); cur = []
        cur.append(line)
    if cur:
        texts.append("\n".join(lead + cur))
    return texts


def chunk_item(blocks: list[Block], ticker: str, company: str, fiscal_year: int, item: str,
               max_words: int = 170, min_words: int = 40, max_table_words: int = 260) -> list[Passage]:
    out: list[Passage] = []

    def emit(text: str, kind: str) -> None:
        if words(text) >= 8 or kind == "table":
            out.append(Passage(id=f"{ticker}-{fiscal_year}-{item}-{len(out) + 1:04d}", ticker=ticker, company=company,
                               fiscal_year=fiscal_year, item=item, kind=kind, text=text, page=None))

    buf: list[str] = []
    heading = ""
    last_short = ""

    def flush() -> None:
        nonlocal heading
        if buf:
            text = " ".join(buf)
            emit((heading + ". " if heading else "") + text, "text")
            buf.clear()
            heading = ""

    for b in blocks:
        if b.kind == "table":
            flush()
            for t in table_passages(b.rows, last_short, max_table_words):
                emit(t, "table")
            last_short = ""
            continue
        text = b.text
        if is_heading(text):
            flush()
            heading = text
            last_short = text
            continue
        last_short = text if words(text) <= 25 else ""
        pieces = split_long(text, max_words) if words(text) > max_words else [text]
        for p in pieces:
            if buf and words(" ".join(buf)) + words(p) > max_words and words(" ".join(buf)) >= min_words:
                flush()
            buf.append(p)
    flush()
    return out
