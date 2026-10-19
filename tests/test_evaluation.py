import numpy as np
import pandas as pd
import pytest

from evaluation import metrics as M
from evaluation.baselines import hybrid, single_signal
from evaluation.dataset import Query, build_queries, catalog_ratings, temporal_split

# --- Métriques : cas calculés à la main ---


def test_precision_recall_hit():
    ranked, rel = [5, 1, 7, 2], {1, 2, 9}
    assert M.precision_at_k(ranked, rel, 4) == pytest.approx(2 / 4)
    assert M.recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)
    assert M.hit_rate_at_k(ranked, rel, 1) == 0.0
    assert M.hit_rate_at_k(ranked, rel, 2) == 1.0
    assert M.recall_at_k(ranked, set(), 4) == 0.0


def test_mrr():
    assert M.mrr_at_k([5, 1, 7], {7}, 10) == pytest.approx(1 / 3)
    assert M.mrr_at_k([5, 1, 7], {7}, 2) == 0.0


def test_ndcg_hand_computed():
    # Pertinents en positions 1 et 3, 2 pertinents au total
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert M.ndcg_at_k([1, 0, 2], {1, 2}, 10) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k([1, 2], {1, 2}, 10) == pytest.approx(1.0)
    assert M.ndcg_at_k([0], set(), 10) == 0.0


def test_diversity_and_coverage():
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert M.intra_list_diversity([0, 1], emb) == pytest.approx(0.0)
    assert M.intra_list_diversity([0, 2], emb) == pytest.approx(1.0)
    assert M.intra_list_diversity([0], emb) == 0.0
    assert M.catalog_coverage([[0, 1], [1, 2]], 10) == pytest.approx(0.3)


def test_bootstrap_ci_contains_mean_and_is_degenerate_for_constants():
    vals = np.array([1.0, 1.0, 1.0, 1.0])
    mean, lo, hi = M.bootstrap_ci(vals, np.array([1, 1, 2, 2]), n_boot=50)
    assert mean == lo == hi == 1.0
    rng = np.random.default_rng(0)
    vals = rng.random(200)
    mean, lo, hi = M.bootstrap_ci(vals, np.repeat(np.arange(50), 4), n_boot=200)
    assert lo <= mean <= hi


# --- Jeu d'évaluation : pas de fuite ---


@pytest.fixture
def ratings():
    rows = []
    for user in (1, 2):
        for t, movie in enumerate([1, 2, 3, 5, 6]):
            rows.append({"userId": user, "movieId": movie, "rating": 4.5 if movie != 3 else 2.0, "timestamp": t})
    rows.append({"userId": 1, "movieId": 999, "rating": 5.0, "timestamp": 99})  # hors catalogue
    return pd.DataFrame(rows)


def test_catalog_ratings_drops_unknown_movies(ratings):
    ml_ids = pd.Series(pd.array([1, 2, 3, None, 5, 6], dtype="Int64"))
    r = catalog_ratings(ratings, ml_ids)
    assert 999 not in set(r["movieId"])
    assert dict(zip(r["movieId"], r["idx"], strict=False))[5] == 4


def test_temporal_split_keeps_oldest_in_train(ratings):
    ml_ids = pd.Series(pd.array([1, 2, 3, None, 5, 6], dtype="Int64"))
    train, test = temporal_split(catalog_ratings(ratings, ml_ids))
    for user in (1, 2):
        tr, te = train[train.userId == user], test[test.userId == user]
        assert len(tr) == 4 and len(te) == 1
        assert tr["timestamp"].max() < te["timestamp"].min()


def test_queries_use_train_likes_and_test_relevance(ratings):
    ml_ids = pd.Series(pd.array([1, 2, 3, None, 5, 6], dtype="Int64"))
    train, test = temporal_split(catalog_ratings(ratings, ml_ids))
    queries = build_queries(train, test, np.array([1, 2]), per_user=2)
    assert queries
    for q in queries:
        assert q.relevant == frozenset({5})  # film 6 (idx 5) : seule note test, aimée
        assert q.idx in q.seen and q.relevant.isdisjoint(q.seen)
    assert [q.idx for q in queries if q.user == 1] == [4, 1]  # aimés du train, plus récents d'abord


# --- Baselines ---


def test_collab_only_has_nothing_to_rank_in_cold_start(art):
    q = Query(user=1, idx=0, relevant=frozenset({1}), seen=frozenset())
    S = art.similarities(0)
    m = single_signal("collab", "c")
    assert m.score(q, S, cold=True).sum() == 0
    np.testing.assert_allclose(m.score(q, S, cold=False), art.similarity("collab", 0))


def test_hybrid_method_matches_engine(art):
    from cinematch.engine import scores

    q = Query(user=1, idx=2, relevant=frozenset({1}), seen=frozenset())
    m = hybrid(art, "h")
    np.testing.assert_allclose(m.score(q, art.similarities(2), cold=False), scores(art, 2), atol=1e-5)
