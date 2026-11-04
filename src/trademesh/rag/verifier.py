"""Deterministic answer verifier (proposal 9.8). No LLM involved.

Checks, in order:
1. every cited label [Pi] exists in the context (no invented citations);
2. the answer cites at least one passage, and we list sentences with no citation;
3. every number in the answer occurs in a cited passage, after normalising $, commas, %,
   and the words thousand/million/billion/trillion. A table that says "(in millions)" also
   makes its plain numbers count at that scale (383,285 -> 383.285 billion).

Failures are returned and shown in the trust panel; they are never hidden or auto-fixed.
Exact matching is deliberate: the prompt asks the model to copy figures verbatim, so a
rounded figure ("$383.3 billion" for 383,285 million) is flagged, which is a known strictness.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..types import Passage
from .prompt import is_insufficient

SCALE = {"thousand": 1e3, "million": 1e6, "mn": 1e6, "mm": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}
TABLE_SCALE_HINT = {"in thousands": 1e3, "in millions": 1e6, "in billions": 1e9}

CITE_RE = re.compile(r"\[P(\d+)\]")
# number, optional decimals, then an optional unit word. Not part of a longer word (1A, 10-K).
NUM_RE = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\d)(?![\w]*-[A-Za-z])(?![A-Za-z])"
    r"(?:\s*(thousand|million|billion|trillion|bn|mn|mm)\b)?"
)


@dataclass
class NumberCheck:
    raw: str
    value: float
    grounded_in: list[int]       # cited labels that contain this value
    in_uncited: list[int]        # context labels that contain it but are not cited


@dataclass
class VerifierResult:
    cited: list[int] = field(default_factory=list)
    invalid_cites: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    numbers: list[NumberCheck] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def ungrounded(self) -> list[NumberCheck]:
        return [n for n in self.numbers if not n.grounded_in]

    @property
    def passed(self) -> bool:
        return not self.flags


def extract_numbers(text: str) -> list[tuple[str, float]]:
    """(raw string, normalised value) for every number in `text`."""
    text = CITE_RE.sub(" ", text)
    out = []
    for m in NUM_RE.finditer(text):
        whole, frac, unit = m.group(1), m.group(2) or "", m.group(3)
        value = float((whole + frac).replace(",", ""))
        if unit:
            value *= SCALE[unit.lower()]
        out.append((m.group(0).strip(), value))
    return out


def passage_values(p: Passage) -> set[float]:
    """All values a reader could take from the passage, incl. table scale and metadata."""
    vals = {float(p.fiscal_year)}
    lower = p.text.lower()
    hint = next((s for h, s in TABLE_SCALE_HINT.items() if h in lower), None)
    for _, v in extract_numbers(p.text):
        vals.add(v)
        if hint:
            vals.add(v * hint)
    # a number written with a unit ("1.2 billion") is also found when quoted bare ("1.2")
    for m in NUM_RE.finditer(p.text):
        if m.group(3):
            vals.add(float((m.group(1) + (m.group(2) or "")).replace(",", "")))
    return vals


def _has(values: set[float], v: float) -> bool:
    return any(math.isclose(v, x, rel_tol=1e-9, abs_tol=1e-9) for x in values)


def verify(answer: str, passages: list[Passage]) -> VerifierResult:
    res = VerifierResult()
    if is_insufficient(answer):
        return res  # an abstention makes no claims, nothing to check
    n = len(passages)
    cited = [int(i) for i in CITE_RE.findall(answer)]
    res.cited = sorted(set(cited))
    res.invalid_cites = [i for i in res.cited if not 1 <= i <= n]
    if res.invalid_cites:
        res.flags.append("invented citation(s): " + ", ".join(f"[P{i}]" for i in res.invalid_cites))
    if not res.cited:
        res.flags.append("answer contains no citation")

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    res.uncited_sentences = [s for s in sentences if not CITE_RE.search(s)]
    if res.cited and res.uncited_sentences:
        res.flags.append(f"{len(res.uncited_sentences)} sentence(s) without a citation")

    values = {i: passage_values(p) for i, p in enumerate(passages, 1)}
    valid_cited = [i for i in res.cited if i not in res.invalid_cites]
    for raw, v in extract_numbers(answer):
        in_cited = [i for i in valid_cited if _has(values[i], v)]
        in_other = [i for i in values if i not in valid_cited and _has(values[i], v)]
        res.numbers.append(NumberCheck(raw, v, in_cited, in_other))
    bad = res.ungrounded
    if bad:
        res.flags.append("number(s) not found in cited passages: " + ", ".join(n.raw for n in bad))
    return res
