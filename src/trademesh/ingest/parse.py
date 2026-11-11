"""10-K HTML -> ordered blocks -> Items (Role A).

A 10-K main document is (inline-XBRL) HTML: paragraphs in <div>/<p>, tables in <table>, a table of
contents near the top that repeats every "Item" heading, and a hidden <ix:header> block of XBRL facts.

Steps
  1. drop scripts, styles, hidden XBRL header and display:none elements
  2. walk the tree into blocks: text paragraphs and tables (layout tables with no numbers become text)
  3. find "Item N." heading blocks; the table of contents produces many headings with almost no text
     after them, so for each Item we keep the span that holds the MOST text
This was written against the structure of EDGAR filings and tested on synthetic filings; eval/corpus
quality is checked by the report that trademesh.ingest.corpus writes (missing or tiny Items are flagged).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

BLOCK_TAGS = {"div", "p", "li", "ul", "ol", "table", "section", "article", "body", "html", "center",
              "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "tr", "td", "th", "main"}
SKIP_TAGS = {"script", "style", "head", "title", "meta", "link"}
NUMERIC_CELL = re.compile(r"^[\(\-–—]?\$?\s*[\d][\d,]*(\.\d+)?\s*\)?%?\)?$")
ITEM_HEADING = re.compile(r"^\s*item\s*(\d{1,2}\s?[abc]?)\s*[\.\:\-–—]?\s*(.*)$", re.I)
_WS = re.compile(r"\s+")


@dataclass
class Block:
    kind: str                       # "text" | "table"
    text: str = ""                  # text blocks: the paragraph
    rows: list[list[str]] = field(default_factory=list)       # table blocks: cleaned cells per row


def clean(s: str) -> str:
    return _WS.sub(" ", s.replace("\xa0", " ").replace("​", "").replace(" ", " ")).strip()


def _is_hidden(tag: Tag) -> bool:
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def _merge_cells(cells: list[str]) -> list[str]:
    """SEC tables put "$", "(" , ")" and "%" in their own cells; glue them back to the numbers."""
    out: list[str] = []
    carry = ""
    for c in cells:
        if not c:
            continue
        if c in ("$", "(", "($", "$("):
            carry += c
            continue
        if c in (")", "%", ")%", "%)") and out:
            out[-1] += c
            continue
        out.append(carry + c)
        carry = ""
    return out


def table_rows(table: Tag) -> list[list[str]]:
    rows = []
    for tr in table.find_all("tr"):
        cells = _merge_cells([clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])])
        if cells:
            rows.append(cells)
    return rows


def is_data_table(rows: list[list[str]]) -> bool:
    """A real data table has several numeric cells; a layout table (headings, bullets) does not."""
    if sum(bool(re.match(r"(item|part)\s*\d*[a-c]?\b", r[0], re.I)) for r in rows) * 2 >= len(rows):
        return False                                           # table of contents: page numbers are not data
    numeric = sum(bool(NUMERIC_CELL.match(c.replace(" ", ""))) for r in rows for c in r)
    return numeric >= 3 and len(rows) >= 2


def html_to_blocks(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(["ix:header"]):
        t.decompose()
    for t in soup.find_all(list(SKIP_TAGS)):
        t.decompose()
    for t in soup.find_all(_is_hidden):
        t.decompose()
    blocks: list[Block] = []

    def flush(buf: list[str]) -> None:
        text = clean(" ".join(buf))
        buf.clear()
        if text:
            blocks.append(Block("text", text))

    def walk(node: Tag) -> None:
        buf: list[str] = []
        for child in node.children:
            if isinstance(child, NavigableString):
                buf.append(str(child))
            elif not isinstance(child, Tag):
                continue
            elif child.name == "table":
                flush(buf)
                rows = table_rows(child)
                if rows and is_data_table(rows):
                    blocks.append(Block("table", rows=rows))
                else:                                     # layout table: each row becomes a paragraph
                    for r in rows:
                        blocks.append(Block("text", clean(" ".join(r))))
            elif child.name in BLOCK_TAGS:
                flush(buf)
                if child.find(lambda t: isinstance(t, Tag) and (t.name in BLOCK_TAGS)):
                    walk(child)
                else:
                    flush([child.get_text(" ")])
            elif child.name == "br":
                buf.append(" ")
            else:
                buf.append(child.get_text(" "))          # inline element
        flush(buf)

    root = soup.body or soup
    walk(root)
    return blocks


# ----------------------------------------------------------------------------- items

KEEP_ITEMS = ["1", "1A", "1C", "2", "3", "5", "7", "7A", "8", "9A"]
ITEM_NAMES = {"1": "Business", "1A": "Risk Factors", "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
              "5": "Market for Registrant's Common Equity", "7": "Management's Discussion and Analysis",
              "7A": "Quantitative and Qualitative Disclosures About Market Risk",
              "8": "Financial Statements and Supplementary Data", "9A": "Controls and Procedures"}


def heading_item(block: Block) -> str | None:
    if block.kind != "text" or len(block.text) > 200:
        return None
    m = ITEM_HEADING.match(block.text)
    return m.group(1).replace(" ", "").upper() if m else None


def _chars(blocks: list[Block]) -> int:
    return sum(len(b.text) if b.kind == "text" else sum(len(c) for r in b.rows for c in r) for b in blocks)


def split_items(blocks: list[Block], keep: list[str] = KEEP_ITEMS, min_chars: int = 300) -> dict[str, list[Block]]:
    """{item: blocks}. For each Item the span with the most text wins, which skips the table of contents."""
    marks = [(i, heading_item(b)) for i, b in enumerate(blocks)]
    marks = [(i, it) for i, it in marks if it]
    best: dict[str, tuple[int, list[Block]]] = {}
    for n, (i, item) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(blocks)
        span = blocks[i + 1:end]
        size = _chars(span)
        if item not in best or size >= best[item][0]:
            best[item] = (size, span)
    return {it: best[it][1] for it in keep if it in best and best[it][0] >= min_chars}
