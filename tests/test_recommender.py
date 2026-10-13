import numpy as np
import pandas as pd
import pytest

from recommender import BASE_WEIGHTS, SIGNAL_TO_MATRIX, recommend, smart_weights


def make_data(n=6, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": range(100, 100 + n),
        "title": [f"Movie {c}" for c in "ABCDEF"[:n]],
        "year": [2000.0 + i for i in range(n)],
        "director": ["Dir"] * n,
        "genres_list": [["Drama"]] * n,
        "overview": ["..."] * n,
    })
    matrices = {}
    for key in SIGNAL_TO_MATRIX.values():
        m = rng.random((n, n)).astype(np.float16)
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)
        matrices[key] = m
    return df, matrices


def test_base_weights_cover_all_signals():
    assert set(BASE_WEIGHTS) == set(SIGNAL_TO_MATRIX)


def test_smart_weights_rules():
    row = pd.Series({"title": "Alien", "genres_list": ["Horror"]})
    w = smart_weights(row)
    assert w["Vibe"] == pytest.approx(BASE_WEIGHTS["Vibe"] + 0.20)

    row = pd.Series({"title": "Parasite", "genres_list": ["Drama"], "original_language": "ko"})
    assert smart_weights(row)["Origine"] == pytest.approx(0.30)

    row = pd.Series({"title": "Toy Story 3", "genres_list": ["Animation"]})
    assert smart_weights(row)["Public"] == pytest.approx(BASE_WEIGHTS["Public"] + 0.10)


def test_recommend_excludes_source_and_is_sorted():
    df, matrices = make_data()
    recos, _ = recommend(df, matrices, idx=2, k=4)
    assert len(recos) == 4
    assert all(r["id"] != df.iloc[2]["id"] for r in recos)
    scores = [r["score"] for r in recos]
    assert scores == sorted(scores, reverse=True)


def test_recommend_k_larger_than_catalog():
    df, matrices = make_data(n=3)
    recos, _ = recommend(df, matrices, idx=0, k=10)
    assert len(recos) == 2


def test_cold_start_redistributes_collab_weight():
    df, matrices = make_data()
    matrices["collab"][1] = 0
    _, w = recommend(df, matrices, idx=1)
    assert w["Public"] == 0
    assert w["Histoire"] == pytest.approx(BASE_WEIGHTS["Histoire"] + 0.2 / 3)


def test_unknown_year_and_director_are_handled():
    df, matrices = make_data()
    df.loc[3, "year"] = 0.0
    df.loc[3, "director"] = ""
    matrices["sbert"][2, 3] = matrices["sbert"][3, 2] = 1.0  # force le film 3 dans le top
    recos, _ = recommend(df, matrices, idx=2, k=5)
    r = next(r for r in recos if r["id"] == df.iloc[3]["id"])
    assert r["year"] is None and r["director"] == ""
