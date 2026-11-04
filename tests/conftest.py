import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # lets tests import the eval/ scripts

from trademesh.data import FIXTURES, load_companies, load_passages  # noqa: E402


@pytest.fixture(scope="session")
def passages():
    return load_passages(FIXTURES / "passages_fixture.jsonl")


@pytest.fixture(scope="session")
def companies():
    return load_companies(FIXTURES / "companies_fixture.json")
