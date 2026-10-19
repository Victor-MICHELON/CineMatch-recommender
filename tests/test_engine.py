import numpy as np
import pandas as pd
import pytest

from cinematch.config import BASE_WEIGHTS, RULES, SIGNALS, WeightConfig
from cinematch.engine import (
    active_rules,
    contributions,
    mmr,
    query_weights,
    recommend,
    redistribute_cold_start,
    smart_weights,
)


def row(**kw):
    base = {"title": "Film", "genres_list": ["Drama"], "original_language": "en", "vote_count": 10}
    return pd.Series({**base, **kw})


def test_base_weights_cover_all_signals():
    assert set(BASE_WEIGHTS) == set(SIGNALS)


@pytest.mark.parametrize(
    ("kwargs", "rule", "signal", "delta"),
    [
        ({"genres_list": ["Animation"]}, "franchise", "keywords", 0.15),
        ({"title": "Toy Story 3"}, "franchise", "collab", 0.10),
        ({"genres_list": ["Horror"]}, "horror", "vibe", 0.20),
        ({"original_language": "ko"}, "foreign", "origin", 0.30),
        ({"vote_count": 3001}, "popular", "collab", 0.15),
    ],
)
def test_each_rule_triggers(kwargs, rule, signal, delta):
    r = row(**kwargs)
    assert rule in active_rules(r)
    assert smart_weights(r)[signal] == pytest.approx(BASE_WEIGHTS[signal] + delta)


def test_no_rule_for_plain_english_film():
    assert active_rules(row()) == []
    assert smart_weights(row()) == pytest.approx(BASE_WEIGHTS)


def test_popular_threshold_is_strict_and_missing_values_are_safe():
    assert "popular" not in active_rules(row(vote_count=3000))
    assert active_rules(row(vote_count=None, original_language=None)) == []


def test_disabled_rule_is_ignored():
    cfg = WeightConfig().without_rules("foreign")
    assert "foreign" not in active_rules(row(original_language="ko"), cfg)
    assert WeightConfig().without_rules().enabled_rules == frozenset()


def test_weights_never_negative():
    cfg = WeightConfig().with_base(dict.fromkeys(SIGNALS, 0.0))
    w = smart_weights(row(genres_list=["Horror"], original_language="fr"), cfg)
    assert min(w.values()) >= 0


def test_cold_start_redistributes_the_actual_collab_weight():
    # Avec la règle « popular », collab vaut 0,35 : c'est ce poids réel qui est redistribué
    w = smart_weights(row(vote_count=10_000))
    assert w["collab"] == pytest.approx(0.35)
    cold = redistribute_cold_start(w)
    assert cold["collab"] == 0
    for s in ("sbert", "keywords", "vibe"):
        assert cold[s] == pytest.approx(w[s] + 0.35 / 3)
    assert sum(cold.values()) == pytest.approx(sum(w.values()))


def test_query_weights_applies_cold_start_only_without_collab(art):
    assert query_weights(art, 3)["collab"] == 0  # film 3 sans donnée collaborative
    assert query_weights(art, 0)["collab"] > 0


def test_override_weights_replace_rules(art):
    w = query_weights(art, 4, override={"credits": 1.0})  # film 4 : non anglophone
    assert w["credits"] == 1.0
    assert w["origin"] == 0.0


def test_contributions_sum_to_score(art):
    res = recommend(art, 0, k=3)
    for item in res.items:
        assert sum(item.contributions.values()) == pytest.approx(item.score, abs=1e-5)


def test_contributions_zero_weights(art):
    assert not contributions(art, 0, dict.fromkeys(SIGNALS, 0.0)).any()


def test_recommend_excludes_source_and_is_sorted(art):
    res = recommend(art, 2, k=4)
    assert len(res.items) == 4
    assert all(r.idx != 2 for r in res.items)
    scores = [r.score for r in res.items]
    assert scores == sorted(scores, reverse=True)
    assert "horror" in res.rules


def test_recommend_k_larger_than_catalog(art):
    assert len(recommend(art, 0, k=50).items) == art.n - 1


def test_multi_query_excludes_all_queries_and_averages(art):
    res = recommend(art, [0, 1], k=10)
    assert {r.idx for r in res.items}.isdisjoint({0, 1})
    single = [recommend(art, q, k=10) for q in (0, 1)]
    target = res.items[0].idx
    expected = np.mean([next((r.score for r in s.items if r.idx == target), 0.0) for s in single])
    assert res.items[0].score == pytest.approx(expected, abs=1e-5)


def test_empty_query_rejected(art):
    with pytest.raises(ValueError):
        recommend(art, [])


def test_mask_filters_candidates(art):
    mask = np.zeros(art.n, dtype=bool)
    mask[[5, 6]] = True
    assert {r.idx for r in recommend(art, 0, k=4, mask=mask).items} == {5, 6}
    assert recommend(art, 0, k=4, mask=np.zeros(art.n, dtype=bool)).items == []


def test_cold_start_flag(art):
    assert recommend(art, 3).cold_start
    assert not recommend(art, 0).cold_start


def test_diversity_zero_keeps_ranking_and_full_diversity_changes_it(art):
    plain = [r.idx for r in recommend(art, 0, k=4).items]
    assert [r.idx for r in recommend(art, 0, k=4, diversity=0.0).items] == plain
    diverse = recommend(art, 0, k=4, diversity=1.0).items
    assert len(diverse) == 4 and len({r.idx for r in diverse}) == 4


def test_mmr_penalizes_near_duplicates():
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    rel = np.array([1.0, 0.99, 0.5], dtype=np.float32)
    assert mmr(np.array([0, 1, 2]), rel, emb, k=2, diversity=0.0) == [0, 1]
    assert mmr(np.array([0, 1, 2]), rel, emb, k=2, diversity=0.7) == [0, 2]


def test_top_signals_and_metadata(art):
    item = recommend(art, 0, k=1).items[0]
    top = item.top_signals(2)
    assert len(top) <= 2 and set(top) <= set(SIGNALS)
    assert item.year == 2000 + item.idx
    assert item.director == art.catalog.iloc[item.idx]["director"]


def test_unknown_year_is_none(art):
    art.catalog.loc[5, "year"] = 0
    res = recommend(art, 4, k=art.n)
    assert next(r for r in res.items if r.idx == 5).year is None


def test_weight_config_roundtrip(tmp_path):
    cfg = WeightConfig().with_base(dict.fromkeys(SIGNALS, 0.5)).without_rules("horror")
    path = tmp_path / "w.json"
    import json

    path.write_text(json.dumps(cfg.to_dict()))
    loaded = WeightConfig.from_json(path)
    assert loaded == cfg
    assert set(loaded.enabled_rules) == set(RULES) - {"horror"}
