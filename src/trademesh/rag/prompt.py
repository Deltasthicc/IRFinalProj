"""RAG prompt and context builder (proposal 9.8).

The context is the top-5 passages labelled [P1]..[P5] with ticker, fiscal year and Item.
The model must answer only from them, cite every claim, and otherwise output exactly
INSUFFICIENT EVIDENCE. Closed-book (the baseline) uses the same model with no passages.
"""
from __future__ import annotations

from ..types import Passage

INSUFFICIENT = "INSUFFICIENT EVIDENCE"

RAG_SYSTEM = f"""You answer questions about SEC 10-K filings using ONLY the numbered passages provided.
Rules:
1. Use only facts that are stated in the passages. Do not use outside knowledge.
2. After every claim write the label of the passage that supports it, for example [P2]. Several labels are allowed: [P1][P3].
3. Copy every number exactly as written in the passage (same digits, same unit). Do not compute, round or convert.
4. Answer in one to four short sentences.
5. If the passages do not contain enough information to answer the question, output exactly: {INSUFFICIENT} (and nothing else)."""

CLOSED_BOOK_SYSTEM = (
    "You answer questions about the SEC 10-K filings of large US companies from your own knowledge. "
    "Answer in one to four short sentences. If you do not know, say so."
)


def label(i: int) -> str:
    return f"[P{i}]"


def build_context(passages: list[Passage]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        header = f"{label(i)} {p.ticker} | FY{p.fiscal_year} | Item {p.item} | {p.kind}"
        blocks.append(f"{header}\n{p.text.strip()}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, passages: list[Passage]) -> str:
    return f"PASSAGES\n{build_context(passages)}\n\nQUESTION: {question.strip()}\n\nANSWER:"


def build_closed_book_prompt(question: str) -> str:
    return f"QUESTION: {question.strip()}\n\nANSWER:"


def is_insufficient(answer: str) -> bool:
    """True if the model abstained. We accept trailing punctuation but nothing else."""
    return answer.strip().strip(".!").strip().upper() == INSUFFICIENT
