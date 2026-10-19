"""L'app démarre sur des artefacts synthétiques et la recherche renvoie 4 cartes."""

import re

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app(artifacts_dir, monkeypatch):
    monkeypatch.setenv("CINEMATCH_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.secrets["TMDB_API_KEY"] = ""
    return at.run()


def cards(at):
    return [m.value for m in at.markdown if '<div class="cm-card">' in m.value]


def search(at, *titles):
    box = at.multiselect(key="queries")
    for title in titles:
        box = box.select(next(i for i, o in enumerate(box.options) if o.startswith(title)))
    box.run()
    at.button[0].click().run()
    return at


def test_app_starts_with_empty_state(app):
    assert not app.exception
    assert any("cm-empty" in m.value for m in app.markdown)
    assert cards(app) == []


def test_search_returns_four_cards_with_explanations(app):
    at = search(app, "Movie A")
    assert not at.exception
    found = cards(at)
    assert len(found) == 4
    assert all(re.search(r"Why : <b>.+</b>", c) for c in found)
    assert all('alt="' in c for c in found)


def test_more_results_and_pagination(app):
    at = search(app, "Movie A")
    at.segmented_control(key="per_page").set_value(8).run()
    assert len(cards(at)) == 7  # 8 films moins le film requête
    at.segmented_control(key="per_page").set_value(4).run()
    next(b for b in at.button if "Next" in b.label).click().run()
    assert len(cards(at)) == 3


def test_language_switch_keeps_search(app):
    at = search(app, "Movie A")
    at.radio(key="lang").set_value("fr").run()
    assert len(cards(at)) == 4
    assert all("Pourquoi" in c for c in cards(at))


def test_cold_start_and_rules_are_explained(app):
    at = search(app, "Movie D")  # Animation, sans donnée collaborative
    captions = " ".join(c.value for c in at.caption)
    assert "animation / sequel" in captions
    assert "No audience data" in captions


def test_filters_can_empty_results(app):
    at = search(app, "Movie A")
    at.multiselect(key="f_genres").select("Horror").run()
    assert len(cards(at)) == 1
    at.multiselect(key="f_langs").select("fr").run()
    assert cards(at) == []
    assert at.warning
