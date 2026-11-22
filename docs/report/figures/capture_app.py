"""Take the application screenshots used in the report (needs the app running on localhost:8501 and Playwright).

    python docs/report/figures/capture_app.py

Uses the committed answer cache, so the Ask example is answered without a live model call.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
URL = "http://localhost:8501"


def shot(page, path: Path) -> None:
    """Full page, then keep only the working area below the tab bar (the answer or the results)."""
    from PIL import Image
    page.screenshot(path=str(path), full_page=True)
    im = Image.open(path)
    im.crop((330, 700, 1470, min(im.height, 2300))).save(path)


def main() -> None:
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1500, "height": 2600})
        page.goto(URL, wait_until="domcontentloaded")
        page.get_by_role("button", name="Ask with a typo").first.wait_for(timeout=300_000)
        page.get_by_role("button", name="Ask with a typo").first.click()
        page.get_by_text("Tesla", exact=False).first.wait_for(timeout=180_000)
        page.wait_for_timeout(8_000)
        page.evaluate("window.scrollTo(0, 0)")
        el = page.locator("div[role='tablist']").first
        el.scroll_into_view_if_needed()
        page.wait_for_timeout(1_500)
        shot(page, OUT / "app_ask.png")
        page.get_by_role("tab", name="Search").click()
        page.wait_for_timeout(3_000)
        box = page.get_by_role("textbox").first
        box.fill("supplier concentration")
        box.press("Enter")
        page.wait_for_timeout(8_000)
        shot(page, OUT / "app_search.png")
        b.close()
    print("saved app_ask.png, app_search.png")


if __name__ == "__main__":
    sys.exit(main())
