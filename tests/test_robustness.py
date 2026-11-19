"""Hostile and odd input: nothing may crash, and nothing user-typed may reach the page as live HTML."""
import random
import string
from pathlib import Path

import pytest

from trademesh.app.backend import load_state
from trademesh.rag.pipeline import RagConfig

APP = Path(__file__).resolve().parents[1] / "src" / "trademesh" / "app" / "streamlit_app.py"

ODD = ["", "   ", "?", "the of and", "a" * 3000, "Apple " * 200, "‮\u0000 weird \U0001F600 emoji", "<script>alert(1)</script>",
       "'; DROP TABLE x;--", "$$$ $5 $5.5.5 1,,2,3 ,,,", "AAPL MSFT TSLA", "Compare Apple", "Compare Apple and Apple", "KO BA GS",
       "\n\n\t", "what is 10-K Item 1A?", "日本語のテスト", "%s %d {0} {x}", "[P1] [P99] INSUFFICIENT EVIDENCE"]


@pytest.fixture(autouse=True)
def _fresh_app_state():
    """The app keeps its search indexes and LLM client in st.cache_resource, which survives from one AppTest to the next
    in the same process. Clear it before and after, so these tests neither inherit nor leak a client."""
    st = pytest.importorskip("streamlit")
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture()
def pipe(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRADEMESH_DATA", str(tmp_path))          # the small fixture: fast
    return load_state().pipeline("bm25", RagConfig(), use_reranker=False)


def test_odd_and_random_questions_never_crash(pipe):
    rng = random.Random(7)
    queries = ODD + ["".join(rng.choice(string.printable) for _ in range(rng.randint(1, 100))) for _ in range(40)]
    for q in queries:
        for kw in ({}, {"compare": True}):
            res = pipe.answer(q, **kw)
            assert isinstance(res.answer, str) and isinstance(res.abstained, bool)
        pipe.matcher.find_mentions(q)
        pipe.matcher.match_string(q)


def test_typed_html_never_reaches_the_page_as_markup(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRADEMESH_DATA", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    evil = '<img src=x onerror=alert(1)> Tesla <script>alert(2)</script> $5 and $6'
    at.text_input(key="ask_q").set_value(evil).run()
    at.text_input(key="search_q").set_value(evil).run()
    at.text_input(key="typo_q").set_value(evil).run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "<img src=x" not in page and "<script>alert" not in page
    assert "onerror" not in page.replace("&lt;img src=x onerror", "")      # only ever as escaped text


def test_offline_cache_miss_is_a_message_not_a_crash(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_PROVIDER", "gemini")                        # a real provider, but offline: no call may be made
    monkeypatch.setenv("LLM_OFFLINE", "1")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setenv("TRADEMESH_DATA", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    at.text_input(key="ask_q").set_value("What does Tesla say about supplier concentration?").run()
    assert not at.exception
    assert any("not in the answer cache" in e.value for e in at.error)
