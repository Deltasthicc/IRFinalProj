"""Company entity detection with spelling tolerance (proposal 9.6, Lecture 3).

Everything is written from scratch, no fuzzy-matching library:

1. The dictionary holds names, tickers and aliases for each company. Each string is
   normalised to a key: lower-case letters and digits only ("Coca-Cola" -> "cocacola").
2. A bigram index maps every character bigram of "$key$" to the entries that contain it.
3. For a query string we collect entries sharing at least one bigram, rank them by
   Jaccard overlap  J(A, B) = |A & B| / |A | B|  of the bigram sets, and keep a short list.
4. Levenshtein edit distance confirms a candidate: at most 2 edits, and at most 1 when
   either string is short (4 characters or fewer, which covers tickers).

Two guards keep false positives down when scanning free text (find_mentions):
* a word that already occurs in the corpus vocabulary is never "corrected" (so "apply"
  is not turned into "Apple"); a typo is a word we have never seen;
* tickers of 3 letters or fewer only match when typed in capitals ("KO", "BA", "GS").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..text import STOPWORDS

MAX_SPAN_TOKENS = 4          # longest company name we look for, in words
MIN_FUZZY_LEN = 3            # tokens shorter than this must match exactly
SHORT_LEN = 4                # <= this many characters: allow at most 1 edit


def normalize(s: str) -> str:
    """Key used for matching: lower-case alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def bigrams(key: str) -> set[str]:
    padded = f"${key}$"
    return {padded[i:i + 2] for i in range(len(padded) - 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def levenshtein(a: str, b: str, limit: int | None = None) -> int:
    """Edit distance with unit-cost insert, delete, substitute (two-row DP).

    If `limit` is given we stop early once every cell of a row exceeds it and
    return limit + 1, which is all the caller needs to reject the pair.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if limit is not None and min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def max_edits(x: str, y: str) -> int:
    return 1 if min(len(x), len(y)) <= SHORT_LEN else 2


@dataclass(frozen=True)
class Entry:
    key: str
    ticker: str
    kind: str       # "name" | "alias" | "ticker"
    surface: str    # the original string, shown to the user ("Did you mean ...")


@dataclass(frozen=True)
class Match:
    ticker: str
    surface: str
    distance: int
    jaccard: float


@dataclass(frozen=True)
class Mention:
    start: int       # character offsets in the original query
    end: int
    text: str
    match: Match

    @property
    def ticker(self) -> str:
        return self.match.ticker


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'.][A-Za-z0-9]+)*|&")


class EntityMatcher:
    def __init__(self, companies: list[dict], known_words: set[str] | None = None,
                 shortlist: int = 10, min_jaccard: float = 0.2):
        self.known_words = known_words or set()
        self.shortlist = shortlist
        self.min_jaccard = min_jaccard
        self.names = {c["ticker"]: c["name"] for c in companies}
        self.entries: list[Entry] = []
        seen = set()
        for c in companies:
            strings = [(c["name"], "name"), (c["ticker"], "ticker")] + [(a, "alias") for a in c.get("aliases", [])]
            # also the name without the legal suffix ("Apple Inc." -> "Apple")
            stripped = re.sub(r",?\s+(inc|corp|corporation|co|company|ltd|plc)\.?$", "", c["name"], flags=re.I)
            if stripped != c["name"]:
                strings.append((stripped, "alias"))
            for surface, kind in strings:
                key = normalize(surface)
                if key and (key, c["ticker"]) not in seen:
                    seen.add((key, c["ticker"]))
                    self.entries.append(Entry(key, c["ticker"], kind, surface))
        self.index: dict[str, list[int]] = {}
        for i, e in enumerate(self.entries):
            for g in bigrams(e.key):
                self.index.setdefault(g, []).append(i)
        self.grams = [bigrams(e.key) for e in self.entries]

    # --- core matching of one string -------------------------------------------------

    def _candidates(self, key: str) -> list[tuple[float, int]]:
        """Shortlist of entry ids by bigram Jaccard (step 3)."""
        qg = bigrams(key)
        ids = {i for g in qg for i in self.index.get(g, ())}
        scored = [(jaccard(qg, self.grams[i]), i) for i in ids]
        scored = [s for s in scored if s[0] >= self.min_jaccard]
        scored.sort(reverse=True)
        return scored[: self.shortlist]

    def match_key(self, key: str, allow_fuzzy: bool = True, tickers_ok: bool = True) -> Match | None:
        if not key:
            return None
        best: list[tuple[int, float, Entry]] = []
        for jac, i in self._candidates(key):
            e = self.entries[i]
            if e.kind == "ticker" and not tickers_ok:
                continue
            d = levenshtein(key, e.key, max_edits(key, e.key))
            if d > max_edits(key, e.key) or (d > 0 and not allow_fuzzy):
                continue
            best.append((d, -jac, e))
        if not best:
            return None
        best.sort(key=lambda t: (t[0], t[1], t[2].kind == "ticker"))
        d, njac, e = best[0]
        # two different companies equally close: refuse to guess
        for d2, njac2, e2 in best[1:]:
            if e2.ticker != e.ticker and (d2, njac2) == (d, njac):
                return None
        return Match(e.ticker, e.surface, d, -njac)

    def match_string(self, text: str) -> Match | None:
        """Standalone use: `text` is known to be a company name or ticker (typo test)."""
        return self.match_key(normalize(text))

    # --- scanning free text ----------------------------------------------------------

    def find_mentions(self, query: str) -> list[Mention]:
        toks = [(m.start(), m.end(), m.group()) for m in _TOKEN_RE.finditer(query)]
        out: list[Mention] = []
        i = 0
        while i < len(toks):
            found = None
            for length in range(min(MAX_SPAN_TOKENS, len(toks) - i), 0, -1):
                span = toks[i:i + length]
                text = query[span[0][0]:span[-1][1]]
                m = self._match_span(text, length)
                if m:
                    found = (length, text, m)
                    break
            if found:
                length, text, m = found
                out.append(Mention(toks[i][0], toks[i + length - 1][1], text, m))
                i += length
            else:
                i += 1
        return out

    def _match_span(self, text: str, n_tokens: int) -> Match | None:
        key = normalize(text)
        if len(key) < 2 or (n_tokens == 1 and text.lower() in STOPWORDS):
            return None
        single = n_tokens == 1
        is_caps = text.isupper()
        # tickers are only trusted when typed in capitals and long enough to be distinctive
        short_ticker_ok = is_caps and len(key) >= 2
        exact = self.match_key(key, allow_fuzzy=False, tickers_ok=short_ticker_ok or len(key) >= 4)
        if exact:
            return exact
        if len(key) < MIN_FUZZY_LEN:
            return None
        if single and text.lower() in self.known_words:
            return None  # a real corpus word, not a typo
        return self.match_key(key, allow_fuzzy=True, tickers_ok=is_caps and len(key) >= 3)

    def detect_tickers(self, query: str) -> list[str]:
        """Distinct tickers in order of first mention."""
        return list(dict.fromkeys(m.ticker for m in self.find_mentions(query)))

    def strip_mentions(self, query: str, mentions: list[Mention] | None = None) -> str:
        """The query with company mentions removed, for the topical (retrieval) part."""
        mentions = self.find_mentions(query) if mentions is None else mentions
        out, last = [], 0
        for m in mentions:
            out.append(query[last:m.start])
            last = m.end
        out.append(query[last:])
        return re.sub(r"\s+", " ", "".join(out)).strip(" ,;:-")

    def corrections(self, mentions: list[Mention]) -> list[tuple[str, str]]:
        """(typed text, suggested name) for mentions that were not an exact hit."""
        return [(m.text, self.names[m.ticker]) for m in mentions if m.match.distance > 0]
