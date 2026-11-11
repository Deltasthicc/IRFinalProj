"""Text analysis for the rankers: tokenisation and the normalisation choices we ablate.

Every choice is a switch in `AnalyzerConfig`, so the preprocessing ablation (proposal 3.1,
RQ2; Lecture 2: tokenisation, stop words, stemming) is a grid over configurations, not a rewrite.

Finance-specific decisions
  * Numbers keep their value but lose thousands separators: "383,285" -> "383285", "5.2" stays "5.2".
    A figure typed with or without commas therefore matches. `keep_numbers=False` drops them.
  * Hyphens and apostrophes split words ("Coca-Cola" -> coca, cola; "company's" -> company, s).
  * `add_metadata` prefixes the indexed text with "ticker company" so a passage that only says
    "the Company" can still be found by an entity query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# A compact classic English stop list (about 120 words, in the spirit of the lecture's 25-word list).
STOP_WORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just me more most my no nor
    not now of off on once only or other our out over own same she should so some such than that the
    their theirs them then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your""".split()
)

_TOKEN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|[a-z0-9]+")
_NUMERIC = re.compile(r"^[\d.]+$")


@dataclass(frozen=True)
class AnalyzerConfig:
    stopwords: bool = True
    stem: bool = False
    keep_numbers: bool = True
    add_metadata: bool = False       # index "ticker company text" instead of "text"

    def label(self) -> str:
        bits = ["stop" if self.stopwords else "nostop", "stem" if self.stem else "nostem",
                "num" if self.keep_numbers else "nonum", "meta" if self.add_metadata else "nometa"]
        return "+".join(bits)


@lru_cache(maxsize=1)
def _porter():
    from nltk.stem.porter import PorterStemmer       # pure Python, no data download needed
    return PorterStemmer()


@lru_cache(maxsize=200_000)
def stem_word(word: str) -> str:
    if _NUMERIC.match(word):
        return word
    return _porter().stem(word)


class Analyzer:
    def __init__(self, config: AnalyzerConfig | None = None):
        self.config = config or AnalyzerConfig()

    def __call__(self, text: str) -> list[str]:
        return self.tokens(text)

    def tokens(self, text: str) -> list[str]:
        c = self.config
        out = []
        for raw in _TOKEN.findall(text.lower()):
            tok = raw.replace(",", "")
            if _NUMERIC.match(tok):
                if not c.keep_numbers:
                    continue
            elif c.stopwords and tok in STOP_WORDS:
                continue
            if c.stem:
                tok = stem_word(tok)
            out.append(tok)
        return out

    def passage_text(self, ticker: str, company: str, text: str) -> str:
        return f"{ticker} {company} {text}" if self.config.add_metadata else text
