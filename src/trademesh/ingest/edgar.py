"""A polite client for SEC EDGAR (Role A).

SEC rules we follow: a declared User-Agent with a contact email (env SEC_USER_AGENT), at most 10
requests per second (we cap at 8), and every response cached on disk so nothing is fetched twice.
If the server answers 403 we stop with a clear message; we never rotate agents or work around it.

Endpoints used (all documented by the SEC):
  https://www.sec.gov/files/company_tickers.json                       ticker -> CIK
  https://data.sec.gov/submissions/CIK##########.json                  filing list of one company
  https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<doc>     the filing's main document
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


class EdgarError(RuntimeError):
    pass


@dataclass
class Filing:
    ticker: str
    cik: int
    accession: str          # with dashes, as EDGAR lists it
    primary_document: str
    report_date: str        # YYYY-MM-DD, end of the fiscal period
    filing_date: str
    form: str = "10-K"

    @property
    def fiscal_year(self) -> int:
        """Year of the period end. Companies whose year ends in January (Walmart, NVIDIA) call that
        year by the calendar year in which it ends, which matches this rule."""
        return int(self.report_date[:4])

    @property
    def url(self) -> str:
        return DOC_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_document)


class EdgarClient:
    def __init__(self, user_agent: str | None = None, cache_dir: str | Path = "data/raw", max_rps: float = 8.0,
                 session=None, sleep=time.sleep):
        ua = user_agent or os.environ.get("SEC_USER_AGENT", "")
        if "@" not in ua:
            raise EdgarError("Set SEC_USER_AGENT to 'Your Name your.email@example.com' (SEC requires a contact address).")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_gap = 1.0 / max_rps
        self._last = 0.0
        self._sleep = sleep
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.network_calls = 0

    def _path(self, url: str) -> Path:
        return self.cache / (hashlib.sha1(url.encode()).hexdigest()[:16] + "_" + url.rsplit("/", 1)[-1][:60])

    def get(self, url: str, retries: int = 4) -> bytes:
        path = self._path(url)
        if path.exists():
            return path.read_bytes()
        for attempt in range(retries):
            wait = self.min_gap - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            self.network_calls += 1
            r = self.session.get(url, timeout=60)
            if r.status_code == 200:
                path.write_bytes(r.content)
                return r.content
            if r.status_code == 403:
                raise EdgarError(f"403 from {url}. Check SEC_USER_AGENT and your network; this client does not retry or work around a block.")
            if r.status_code in (429, 500, 502, 503, 504):
                self._sleep(2 ** attempt)
                continue
            raise EdgarError(f"{r.status_code} for {url}")
        raise EdgarError(f"gave up on {url} after {retries} attempts")

    def get_json(self, url: str):
        return json.loads(self.get(url))

    def ticker_to_cik(self) -> dict[str, int]:
        data = self.get_json(TICKERS_URL)
        return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}

    def latest_10k(self, ticker: str, ciks: dict[str, int] | None = None) -> Filing:
        ciks = ciks or self.ticker_to_cik()
        cik = ciks.get(ticker.upper())
        if cik is None:
            raise EdgarError(f"unknown ticker {ticker}")
        recent = self.get_json(SUBMISSIONS_URL.format(cik=cik))["filings"]["recent"]
        for i, form in enumerate(recent["form"]):
            if form == "10-K":                              # the list is newest first; 10-K/A is a different form string
                return Filing(ticker.upper(), cik, recent["accessionNumber"][i], recent["primaryDocument"][i],
                              recent["reportDate"][i], recent["filingDate"][i])
        raise EdgarError(f"no 10-K in the recent filings of {ticker}")

    def download_html(self, filing: Filing) -> str:
        return self.get(filing.url).decode("utf-8", errors="replace")
