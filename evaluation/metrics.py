"""Métriques de recommandation top-k et intervalles de confiance bootstrap."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
from numpy.typing import NDArray


def precision_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    return sum(1 for i in ranked[:k] if i in relevant) / k


def recall_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for i in ranked[:k] if i in relevant) / len(relevant)


def hit_rate_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    return float(any(i in relevant for i in ranked[:k]))


def mrr_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    for rank, i in enumerate(ranked[:k], 1):
        if i in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    """NDCG binaire : gain 1 par film pertinent, idéal = min(k, |pertinents|) en tête."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(rank + 1) for rank, i in enumerate(ranked[:k], 1) if i in relevant)
    ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return float(dcg / ideal)


def intra_list_diversity(items: Sequence[int], embeddings: NDArray[np.float32]) -> float:
    """1 − similarité cosinus moyenne entre paires de la liste (embeddings normalisés)."""
    if len(items) < 2:
        return 0.0
    e = embeddings[list(items)]
    sim = e @ e.T
    n = len(items)
    return float(1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))


def catalog_coverage(lists: Sequence[Sequence[int]], n_items: int) -> float:
    """Part du catalogue apparaissant au moins une fois dans les listes."""
    return len({i for lst in lists for i in lst}) / n_items


def bootstrap_ci(
    values: NDArray[np.float64],
    groups: NDArray[np.int64],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Moyenne et IC à (1 − alpha) par bootstrap sur les groupes (utilisateurs).

    Chaque tirage ré-échantillonne les utilisateurs avec remise et prend la
    moyenne de toutes leurs requêtes : les requêtes d'un même utilisateur ne
    sont pas indépendantes.
    """
    values = np.asarray(values, dtype=np.float64)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
