"""Tests d'intégration sur les vrais artefacts (`python -m pipeline.build`) : `pytest -m slow`."""

from pathlib import Path

import pytest

from cinematch import load_artifacts, recommend
from cinematch.config import RULES

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="artefacts absents"),
]


@pytest.fixture(scope="module")
def art():
    return load_artifacts(ARTIFACTS)


def find(art, title, year):
    cat = art.catalog
    return int(cat.index[(cat["title"] == title) & (cat["year"] == year)][0])


def test_artifacts_are_small(art):
    total = sum(f["bytes"] for f in art.manifest["files"].values())
    assert total < 60e6
    assert art.n > 6000


def test_all_four_rules_fire_on_real_films(art):
    fired = set()
    for title, year in [("Parasite", 2019), ("Toy Story", 1995), ("The Conjuring", 2013)]:
        fired |= set(recommend(art, find(art, title, year)).rules)
    assert fired == set(RULES)


def test_parasite_gets_korean_recommendations(art):
    res = recommend(art, find(art, "Parasite", 2019), k=4)
    assert "foreign" in res.rules
    langs = [art.catalog.iloc[r.idx]["original_language"] for r in res.items]
    assert langs.count("ko") >= 3


def test_sequels_are_found(art):
    titles = {r.title for r in recommend(art, find(art, "Toy Story", 1995), k=4).items}
    assert {"Toy Story 2", "Toy Story 3"} <= titles


def test_cold_start_share_matches_readme(art):
    assert 0.40 < 1 - art.collab_coverage < 0.46
