"""Builds the OFFLINE TEST FIXTURE (3 companies, 30 short passages).

EVERYTHING HERE IS INVENTED for unit tests. The names are real companies but the text and
numbers are made up, every passage starts with "[FIXTURE]", and ids start with "FIX-".
Never use these files for any reported result.

Run:  python tests/fixtures/build_fixture.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from trademesh.data import save_passages  # noqa: E402
from trademesh.types import Passage  # noqa: E402

COMPANIES = [
    {"ticker": "AAPL", "name": "Apple Inc.", "aliases": ["Apple", "iPhone maker"]},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "aliases": ["Microsoft", "MSFT Corp"]},
    {"ticker": "TSLA", "name": "Tesla, Inc.", "aliases": ["Tesla"]},
]

# (ticker, item, kind, text). Fiscal year 2025 for all.
RAW = [
    ("AAPL", "1", "text", "Apple designs smartphones, personal computers, tablets and wearables, and sells related services."),
    ("AAPL", "1", "text", "The company sells through its own retail stores, online stores and third-party carriers and resellers."),
    ("AAPL", "1A", "text", "Apple depends on a limited number of outsourcing partners, mostly in Asia, to manufacture its products. Supplier concentration could disrupt supply."),
    ("AAPL", "1A", "text", "Tariffs and trade restrictions could raise product costs and reduce gross margin."),
    ("AAPL", "1A", "text", "Component shortages and long lead times for custom chips may limit production of new products."),
    ("AAPL", "7", "text", "Net sales increased 4 percent during the year, driven by growth in services and higher iPhone sales."),
    ("AAPL", "7", "table", "Net sales by category (in millions). iPhone 200,000; Mac 30,000; iPad 28,000; Services 95,000; Total net sales 400,000."),
    ("AAPL", "7A", "text", "A hypothetical 10 percent change in interest rates would not materially affect the fair value of marketable securities."),
    ("AAPL", "5", "text", "The company repurchased shares of its common stock under a publicly announced program during the fourth quarter."),
    ("AAPL", "2", "text", "The headquarters campus is located in Cupertino, California and is owned by the company."),
    ("MSFT", "1", "text", "Microsoft develops cloud software, productivity applications, gaming and business solutions for customers worldwide."),
    ("MSFT", "1", "text", "The Intelligent Cloud segment includes Azure, server products and enterprise services."),
    ("MSFT", "1A", "text", "Microsoft relies on third-party data center capacity and a limited number of suppliers of advanced chips for AI infrastructure. Supply constraints could affect growth."),
    ("MSFT", "1A", "text", "Cybersecurity incidents and nation-state attacks could harm customers and damage the reputation of the company."),
    ("MSFT", "1A", "text", "Regulatory scrutiny of artificial intelligence may impose new compliance costs and limit product offerings."),
    ("MSFT", "7", "text", "Revenue increased 15 percent, led by growth in Azure and other cloud services."),
    ("MSFT", "7", "table", "Revenue by segment (in millions). Productivity and Business Processes 120,000; Intelligent Cloud 110,000; More Personal Computing 65,000; Total 295,000."),
    ("MSFT", "7A", "text", "Foreign currency risk arises from revenue earned in currencies other than the US dollar; the company uses forward contracts to hedge."),
    ("MSFT", "9", "text", "There were no disagreements with accountants on accounting or financial disclosure."),
    ("MSFT", "3", "text", "The company is subject to various legal proceedings, none of which is expected to be material."),
    ("TSLA", "1", "text", "Tesla designs, manufactures and sells electric vehicles, energy generation and storage systems."),
    ("TSLA", "1", "text", "The company operates a network of Superchargers and sells vehicles directly to customers online and in stores."),
    ("TSLA", "1A", "text", "Tesla relies on a small number of suppliers for battery cells; supplier concentration and raw material shortages such as lithium could disrupt production."),
    ("TSLA", "1A", "text", "Changes to government incentives for electric vehicles could reduce demand for the company's products."),
    ("TSLA", "1A", "text", "The company depends on the services of its Chief Executive Officer, and the loss of his services could harm the business."),
    ("TSLA", "7", "text", "Total revenues decreased 2 percent, mainly due to lower average selling prices for vehicles."),
    ("TSLA", "7", "table", "Revenue by source (in millions). Automotive sales 70,000; Regulatory credits 2,500; Energy generation and storage 10,000; Services and other 12,000; Total revenues 94,500."),
    ("TSLA", "7A", "text", "The company is exposed to foreign currency risk related to its operations in China and Europe."),
    ("TSLA", "2", "text", "Gigafactory sites in Nevada, Texas, Shanghai and Berlin are used for vehicle and battery production."),
    ("TSLA", "3", "text", "The company is party to litigation relating to its driver assistance systems and to employment matters."),
]


def main() -> None:
    passages = []
    counters: dict[tuple, int] = {}
    for ticker, item, kind, text in RAW:
        key = (ticker, item)
        counters[key] = counters.get(key, 0) + 1
        company = next(c["name"] for c in COMPANIES if c["ticker"] == ticker)
        passages.append(
            Passage(
                id=f"FIX-{ticker}-2025-{item}-{counters[key]:04d}",
                ticker=ticker, company=company, fiscal_year=2025, item=item, kind=kind,
                text="[FIXTURE] " + text, page=None,
            )
        )
    save_passages(passages, HERE / "passages_fixture.jsonl")
    (HERE / "companies_fixture.json").write_text(json.dumps(COMPANIES, indent=2), encoding="utf-8")
    print(f"wrote {len(passages)} fixture passages")


if __name__ == "__main__":
    main()
