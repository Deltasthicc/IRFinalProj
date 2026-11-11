"""Ingest tests on a SYNTHETIC filing that imitates EDGAR's HTML structure (TOC, ix:header, split cells).
They show the code does what it is designed to do; they do not prove it parses every real 10-K.
The real-corpus check is the report written by trademesh.ingest.corpus."""
import json

import pytest

from trademesh.ingest.chunk import chunk_item, table_passages
from trademesh.ingest.corpus import build_passages, main
from trademesh.ingest.edgar import EdgarClient, EdgarError, Filing
from trademesh.ingest.parse import Block, html_to_blocks, split_items, table_rows

RISK = " ".join(f"Sentence {i} explains that tariffs and supply chain disruption could harm results." for i in range(60))
BIZ = " ".join(f"The company designs and sells product line {i} to customers worldwide." for i in range(40))
MDA = " ".join(f"Net sales changed in segment {i} because of pricing and volume." for i in range(40))


def synthetic_10k() -> str:
    toc = "".join(f"<tr><td>Item {i}.</td><td>{n}</td><td>{p}</td></tr>"
                  for i, n, p in [("1", "Business", 3), ("1A", "Risk Factors", 8), ("7", "MD&amp;A", 30), ("8", "Financial Statements", 40)])
    return f"""<html><body>
<ix:header><ix:hidden>Hidden XBRL fact that must not appear</ix:hidden></ix:header>
<div>ACME CORP FORM 10-K</div>
<table>{toc}</table>
<div><span>Item 1.</span> <span>Business</span></div>
<div>{BIZ}</div>
<div>Item 1A. Risk Factors</div>
<div>Supply chain</div>
<div>{RISK}</div>
<div>See Item 7. for more.</div>
<div style="display:none">invisible text</div>
<div>Item 7. Management's Discussion and Analysis</div>
<div>Results of Operations</div>
<div>{MDA}</div>
<div>(in millions)</div>
<table>
<tr><td></td><td>2025</td><td></td><td>2024</td></tr>
<tr><td>Net sales</td><td>$</td><td>383,285</td><td>$</td><td>394,328</td></tr>
<tr><td>Cost of sales</td><td>(</td><td>210,352</td><td>)</td><td>(214,137</td><td>)</td></tr>
<tr><td>Growth</td><td>4.2</td><td>%</td><td>2.1</td><td>%</td></tr>
</table>
<div>Item 8. Financial Statements and Supplementary Data</div>
<div>{BIZ}</div>
</body></html>"""


def test_blocks_skip_hidden_and_merge_split_cells():
    blocks = html_to_blocks(synthetic_10k())
    text = " ".join(b.text for b in blocks if b.kind == "text")
    assert "Hidden XBRL" not in text and "invisible text" not in text
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1                                   # the TOC (no numbers) became text, the data table stayed a table
    rows = tables[0].rows
    assert rows[1] == ["Net sales", "$383,285", "$394,328"]
    assert rows[2] == ["Cost of sales", "(210,352)", "(214,137)"]
    assert rows[3] == ["Growth", "4.2%", "2.1%"]


def test_items_pick_the_real_span_not_the_toc():
    items = split_items(html_to_blocks(synthetic_10k()))
    assert set(items) == {"1", "1A", "7", "8"}
    assert "tariffs" in " ".join(b.text for b in items["1A"])
    assert "product line 3" in " ".join(b.text for b in items["1"])           # not the 3-cell TOC row
    assert not any("Item 1A" in b.text and len(b.text) < 40 for b in items["1A"])


def test_build_passages_table_chunk_keeps_labels_and_caption():
    ps, rep = build_passages(synthetic_10k(), "ACME", "Acme Corp", 2025)
    assert rep["passages"] == len(ps) > 5 and rep["table_passages"] == 1
    t = next(p for p in ps if p.kind == "table")
    assert t.item == "7" and "(in millions)" in t.text
    assert "Net sales | $383,285 | $394,328" in t.text and "Columns:" in t.text
    assert ps[0].id == "ACME-2025-1-0001" and len({p.id for p in ps}) == len(ps)
    assert all(p.ticker == "ACME" and p.fiscal_year == 2025 for p in ps)
    assert max(len(p.text.split()) for p in ps if p.kind == "text") <= 200
    assert any(p.text.startswith("Supply chain. ") for p in ps)                # heading carried into the passage
    assert any("Item 1A not found" in w for w in rep["warnings"]) is False
    assert "only" in " ".join(rep["warnings"])                                  # a tiny filing is flagged
    assert any("item X" in n for n in rep["notes"])   # and tries the unsectioned-text fallback


def test_large_table_is_split_with_header_repeated():
    rows = [["Metric", "2025", "2024"]] + [[f"Line {i}", f"{i * 10:,}", f"{i * 9:,}"] for i in range(80)]
    parts = table_passages(rows, "Segment results", 60)
    assert len(parts) > 1
    assert all(p.startswith("Segment results\nColumns: Metric | 2025 | 2024") for p in parts)


def test_long_paragraph_is_split_at_sentences():
    par = " ".join(f"This is sentence number {i} of a very long paragraph." for i in range(100))
    ps = chunk_item([Block("text", par)], "T", "T Inc", 2025, "1")
    assert len(ps) > 3 and all(len(p.text.split()) <= 180 for p in ps)
    assert all(p.text.rstrip().endswith(".") for p in ps)                       # cut at sentence ends only


# ------------------------------------------------------------------ EDGAR client (no network: fake session)

class FakeResp:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, timeout=0):
        self.calls.append(url)
        r = self.routes.get(url)
        return r if r is not None else FakeResp(404)


def client(tmp_path, routes):
    return EdgarClient("Test User test@example.com", cache_dir=tmp_path, session=FakeSession(routes), sleep=lambda s: None)


def test_edgar_requires_contact_email(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(EdgarError):
        EdgarClient(cache_dir=tmp_path)


def test_edgar_finds_latest_10k_and_caches(tmp_path):
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    subs = {"filings": {"recent": {"form": ["8-K", "10-K", "10-K"], "accessionNumber": ["a-1", "0000320193-25-000079", "old"],
                                   "primaryDocument": ["x.htm", "aapl-20250927.htm", "o.htm"],
                                   "reportDate": ["2025-10-30", "2025-09-27", "2024-09-28"],
                                   "filingDate": ["2025-10-30", "2025-10-31", "2024-11-01"]}}}
    routes = {"https://www.sec.gov/files/company_tickers.json": FakeResp(200, json.dumps(tickers).encode()),
              "https://data.sec.gov/submissions/CIK0000320193.json": FakeResp(200, json.dumps(subs).encode()),
              "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm": FakeResp(200, b"<html>x</html>")}
    c = client(tmp_path, routes)
    f = c.latest_10k("aapl")
    assert (f.cik, f.fiscal_year, f.accession) == (320193, 2025, "0000320193-25-000079")
    assert c.download_html(f) == "<html>x</html>"
    n = c.network_calls
    c.download_html(f); c.ticker_to_cik()
    assert c.network_calls == n                                     # everything served from the cache
    with pytest.raises(EdgarError):
        c.latest_10k("NOPE")


def test_edgar_403_is_not_retried(tmp_path):
    c = client(tmp_path, {"https://x/y": FakeResp(403)})
    with pytest.raises(EdgarError, match="403"):
        c.get("https://x/y")
    assert len(c.session.calls) == 1


def test_cli_from_html(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "AAPL_2025.html").write_text(synthetic_10k())
    (raw / "bogus.html").write_text("x")
    out = tmp_path / "out"
    assert main(["--from-html", str(raw), "--out", str(out), "--tickers", "AAPL"]) == 0
    lines = (out / "passages.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["company"] == "Apple Inc."
    assert json.loads((out / "companies.json").read_text())[0]["ticker"] == "AAPL"
    assert "summary" in json.loads((out / "corpus_report.json").read_text())


def test_running_page_headers_do_not_split_an_item_and_split_words_are_rejoined():
    pages = "".join(f'<div>Item 7</div><div>{MDA[i * 400:(i + 1) * 400]}</div>' for i in range(6))
    html = f"""<html><body>
<div>ITEM 6. RESERVED</div><div>{BIZ[:400]}</div>
<div><span>ITEM 7. MANAGEMENT</span><span>'S DISCUSSION</span></div>
{pages}
<div><span>QUALITAT</span><span>IVE</span> disclosures</div>
<div>Item 8</div><div><span>ITEM 8. FINANCIAL STATE</span><span>MENTS</span></div><div>{BIZ}</div>
</body></html>"""
    blocks = html_to_blocks(html)
    items = split_items(blocks)
    text7 = " ".join(b.text for b in items["7"])
    assert "segment 39" in text7 or "segment 3" in text7          # the whole Item survives, not one page of it
    assert len(text7) > 2000
    assert not any(b.text == "Item 7" for b in items["7"])        # running headers are not content
    assert "QUALITATIVE disclosures" in text7
    assert any("FINANCIAL STATEMENTS" in b.text for b in blocks)


def _filing(*parts: str) -> str:
    return "<html><body>" + "".join(f"<div>{p}</div>" for p in parts) + "</body></html>"


def test_financials_under_item_15_are_taken_when_item_8_is_only_a_pointer():
    body = " ".join(f"Note {i} describes revenue recognition and income taxes for the year." for i in range(600))
    html = _filing("Item 1. Business", BIZ * 2, "Item 7. Management's Discussion and Analysis", MDA * 2,
                   "Item 8. Financial Statements and Supplementary Data",
                   "The information required by this Item is set forth in Item 15. of this report, filed herewith.",
                   "Item 15. Exhibit and Financial Statement Schedules", body)
    ps, rep = build_passages(html, "NVDA", "NVIDIA", 2026)
    assert rep["per_item"].get("15", 0) >= 20 and any(p.item == "15" for p in ps)
    assert any("Item 15" in n for n in rep["notes"])


def test_filing_without_item_headings_still_gets_passages():
    html = _filing("FORM 10-K CROSS-REFERENCE INDEX", "Item 1 Business page 3", "Business",
                   *[f"{BIZ} {MDA} {RISK}" for _ in range(6)])
    ps, rep = build_passages(html, "INTC", "Intel", 2025)
    assert len(ps) > 20 and {p.item for p in ps} == {"X"}
    assert any("item X" in n for n in rep["notes"])


def test_edgar_looks_in_older_submission_files(tmp_path):
    tickers = {"0": {"cik_str": 34088, "ticker": "XOM", "title": "Exxon"}}
    empty = {"form": ["4", "4"], "accessionNumber": ["a", "b"], "primaryDocument": ["x", "y"], "reportDate": ["", ""], "filingDate": ["", ""]}
    older = {"form": ["4", "10-K"], "accessionNumber": ["c", "0000034088-26-000010"], "primaryDocument": ["z", "xom-20251231.htm"],
             "reportDate": ["", "2025-12-31"], "filingDate": ["", "2026-02-18"]}
    routes = {"https://www.sec.gov/files/company_tickers.json": FakeResp(200, json.dumps(tickers).encode()),
              "https://data.sec.gov/submissions/CIK0000034088.json": FakeResp(200, json.dumps(
                  {"filings": {"recent": empty, "files": [{"name": "CIK0000034088-submissions-001.json"}]}}).encode()),
              "https://data.sec.gov/submissions/CIK0000034088-submissions-001.json": FakeResp(200, json.dumps(older).encode())}
    f = client(tmp_path, routes).latest_10k("XOM")
    assert (f.accession, f.fiscal_year) == ("0000034088-26-000010", 2025)


def test_xom_uses_the_predecessor_cik_for_its_10k(tmp_path):
    tickers = {"0": {"cik_str": 2115436, "ticker": "XOM", "title": "ExxonMobil Holdings Corp"}}
    subs = {"filings": {"recent": {"form": ["10-K"], "accessionNumber": ["0000034088-26-000010"], "primaryDocument": ["xom.htm"],
                                   "reportDate": ["2025-12-31"], "filingDate": ["2026-02-18"]}}}
    routes = {"https://www.sec.gov/files/company_tickers.json": FakeResp(200, json.dumps(tickers).encode()),
              "https://data.sec.gov/submissions/CIK0000034088.json": FakeResp(200, json.dumps(subs).encode())}
    f = client(tmp_path, routes).latest_10k("XOM")
    assert (f.cik, f.fiscal_year) == (34088, 2025)
