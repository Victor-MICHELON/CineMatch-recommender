"""Jeu d'évaluation item-to-item construit sur MovieLens `ml-latest-small`.

Protocole :
- on ne garde que les notes de films du catalogue (jointure `links.csv`) ;
- découpage temporel par utilisateur : 80 % de notes les plus anciennes en train,
  20 % les plus récentes en test ;
- positif = note ≥ 4 ;
- requête = un film aimé du train (les plus récents d'abord) ; pertinents = les
  films aimés de la partie test du même utilisateur ;
- utilisateurs répartis en validation (réglage des poids) et test (chiffres finaux).

Le signal `collab` est recalculé sur la partie train uniquement (voir `run.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LIKE_THRESHOLD = 4.0


@dataclass(frozen=True)
class Query:
    user: int
    idx: int  # position du film requête dans le catalogue
    relevant: frozenset[int]  # films aimés dans la partie test
    seen: frozenset[int]  # films notés dans le train (exclus du classement)


@dataclass
class EvalData:
    train: pd.DataFrame
    test: pd.DataFrame
    val_queries: list[Query]
    test_queries: list[Query]


def catalog_ratings(ratings: pd.DataFrame, movielens_ids: pd.Series) -> pd.DataFrame:
    """Notes restreintes au catalogue, avec la position `idx` du film."""
    pos = pd.Series(np.arange(len(movielens_ids)), index=movielens_ids.to_numpy())
    pos = pos[pos.index.notna()]
    pos = pos[~pos.index.duplicated()]
    r = ratings[ratings["movieId"].isin(pos.index)].copy()
    r["idx"] = pos.loc[r["movieId"].to_numpy()].to_numpy()
    return r.reset_index(drop=True)


def temporal_split(ratings: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Par utilisateur : les `train_frac` notes les plus anciennes en train, le reste en test."""
    r = ratings.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    rank = r.groupby("userId").cumcount()
    size = r.groupby("userId")["movieId"].transform("size")
    is_train = rank < np.floor(size * train_frac)
    return r[is_train].reset_index(drop=True), r[~is_train].reset_index(drop=True)


def build_queries(train: pd.DataFrame, test: pd.DataFrame, users: np.ndarray, per_user: int) -> list[Query]:
    queries = []
    train_by_user = dict(tuple(train.groupby("userId")))
    test_liked = test[test["rating"] >= LIKE_THRESHOLD].groupby("userId")["idx"].apply(frozenset)
    for user in users:
        relevant = test_liked.get(user)
        tr = train_by_user.get(user)
        if not relevant or tr is None:
            continue
        seen = frozenset(tr["idx"].tolist())
        liked = tr[tr["rating"] >= LIKE_THRESHOLD].sort_values("timestamp", ascending=False)
        for idx in liked["idx"].head(per_user):
            rel = relevant - {int(idx)}
            if rel:
                queries.append(Query(int(user), int(idx), rel, seen))
    return queries


def make_eval_data(
    ratings: pd.DataFrame,
    movielens_ids: pd.Series,
    val_frac: float = 0.2,
    per_user: int = 5,
    seed: int = 42,
) -> EvalData:
    r = catalog_ratings(ratings, movielens_ids)
    train, test = temporal_split(r)
    users = np.array(sorted(r["userId"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_val = round(len(users) * val_frac)
    return EvalData(
        train=train,
        test=test,
        val_queries=build_queries(train, test, users[:n_val], per_user),
        test_queries=build_queries(train, test, users[n_val:], per_user),
    )
