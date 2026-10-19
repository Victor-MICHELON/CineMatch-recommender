"""Méthodes comparées : baselines et variantes du moteur hybride.

Chaque méthode reçoit la requête, la pile des 8 lignes de similarité du film
requête (`S`, 8 × n, calculée une seule fois et partagée par toutes les
méthodes) et un drapeau `cold` (cold start simulé : le signal collaboratif du
film requête est masqué). Elle renvoie un score par film du catalogue.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cinematch.config import SIGNALS, WeightConfig
from cinematch.data import Artifacts
from cinematch.engine import redistribute_cold_start, smart_weights
from evaluation.dataset import Query

ScoreFn = Callable[[Query, NDArray[np.float32], bool], NDArray[np.float32]]
COLLAB = SIGNALS.index("collab")


@dataclass(frozen=True)
class Method:
    name: str
    score: ScoreFn


def popularity(train: pd.DataFrame, n: int) -> Method:
    """Films les plus notés dans le train, identiques pour toutes les requêtes."""
    counts = np.bincount(train["idx"].to_numpy(), minlength=n).astype(np.float32)
    return Method("Popularité", lambda q, S, cold: counts)


def random_baseline(n: int, seed: int = 0) -> Method:
    def score(q: Query, S: NDArray[np.float32], cold: bool) -> NDArray[np.float32]:
        rng = np.random.default_rng([seed, q.user, q.idx])
        return rng.random(n, dtype=np.float32)

    return Method("Aléatoire", score)


def single_signal(signal: str, name: str) -> Method:
    k = SIGNALS.index(signal)

    def score(q: Query, S: NDArray[np.float32], cold: bool) -> NDArray[np.float32]:
        if cold and k == COLLAB:
            return np.zeros(S.shape[1], dtype=np.float32)  # aucune donnée : rien à classer
        return S[k]

    return Method(name, score)


def weight_vector(
    row: pd.Series,
    cfg: WeightConfig,
    cold: bool,
    override: Mapping[str, float] | None = None,
    drop: frozenset[str] = frozenset(),
) -> NDArray[np.float32]:
    w = {s: float(override[s]) for s in SIGNALS} if override else smart_weights(row, cfg)
    if cold:
        w = redistribute_cold_start(w, cfg)
    for s in drop:
        w[s] = 0.0
    return np.array([w[s] for s in SIGNALS], dtype=np.float32)


def hybrid(
    art: Artifacts,
    name: str,
    cfg: WeightConfig | None = None,
    override: Mapping[str, float] | None = None,
    drop: frozenset[str] = frozenset(),
) -> Method:
    """Moteur hybride : mêmes règles et même cold start que `cinematch.engine`."""
    cfg = cfg or WeightConfig()
    rows = art.catalog

    def score(q: Query, S: NDArray[np.float32], cold: bool) -> NDArray[np.float32]:
        w = weight_vector(rows.iloc[q.idx], cfg, cold, override, drop)
        total = float(w.sum())
        return (w @ S) / total if total > 0 else np.zeros(S.shape[1], dtype=np.float32)

    return Method(name, score)


def standard_methods(art: Artifacts, train: pd.DataFrame, tuned: WeightConfig | None = None) -> list[Method]:
    methods = [
        popularity(train, art.n),
        random_baseline(art.n),
        single_signal("sbert", "SBERT seul"),
        single_signal("collab", "Collaboratif seul"),
        hybrid(art, "Hybride, poids uniformes", override=dict.fromkeys(SIGNALS, 1.0)),
        hybrid(art, "Hybride, poids de base sans règles", cfg=WeightConfig().without_rules()),
        hybrid(art, "Hybride actuel (base + règles)"),
    ]
    if tuned is not None:
        methods.append(hybrid(art, "Hybride optimisé (Optuna)", cfg=tuned))
    return methods
