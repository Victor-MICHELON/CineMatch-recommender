"""Moteur de recommandation hybride (sans dépendance à Streamlit).

Pour un film requête `i` et un candidat `j` :

    S(i, j) = Σ_k w_k(i) · M_k[i, j] / Σ_k w_k(i)

Les poids `w_k(i)` partent de `WeightConfig.base` et sont ajustés par des règles
(genre, langue, popularité) puis par la redistribution du cold start. Chaque
terme `w_k · M_k[i, j] / Σ w` est conservé : c'est la contribution du signal `k`,
qui sert à expliquer chaque recommandation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cinematch.config import SIGNALS, WeightConfig
from cinematch.data import Artifacts

DEFAULT_CONFIG = WeightConfig()
MMR_POOL = 50


@dataclass(frozen=True)
class Recommendation:
    idx: int
    id: int
    title: str
    year: int | None
    director: str
    score: float
    contributions: dict[str, float]

    def top_signals(self, n: int = 2) -> list[str]:
        """Les `n` signaux qui ont le plus contribué au score (contribution > 0)."""
        ranked = sorted(self.contributions.items(), key=lambda kv: kv[1], reverse=True)
        return [s for s, c in ranked[:n] if c > 0]


@dataclass(frozen=True)
class RecommendationResult:
    items: list[Recommendation]
    weights: dict[str, float]  # poids effectifs (moyenne sur les films requêtes)
    rules: list[str]  # règles déclenchées par au moins un film requête
    cold_start: bool  # au moins un film requête sans donnée collaborative


def active_rules(row: pd.Series, cfg: WeightConfig = DEFAULT_CONFIG) -> list[str]:
    """Règles adaptatives déclenchées par un film (sous-ensemble de `cfg.enabled_rules`)."""
    genres = set(row.get("genres_list") or [])
    title = str(row.get("title", ""))
    lang = row.get("original_language")
    votes = row.get("vote_count")
    triggered = {
        "franchise": "Animation" in genres or any(ch.isdigit() for ch in title),
        "horror": "Horror" in genres,
        "foreign": isinstance(lang, str) and lang not in ("", "en"),
        "popular": votes is not None and not pd.isna(votes) and float(votes) > cfg.popular_min_votes,
    }
    return [r for r, on in triggered.items() if on and r in cfg.enabled_rules]


def smart_weights(row: pd.Series, cfg: WeightConfig = DEFAULT_CONFIG) -> dict[str, float]:
    """Poids de base ajustés par les règles déclenchées par le film (≥ 0)."""
    w = {s: float(cfg.base.get(s, 0.0)) for s in SIGNALS}
    for rule in active_rules(row, cfg):
        for signal, delta in cfg.rule_deltas(rule).items():
            w[signal] += delta
    return {s: max(v, 0.0) for s, v in w.items()}


def redistribute_cold_start(weights: Mapping[str, float], cfg: WeightConfig = DEFAULT_CONFIG) -> dict[str, float]:
    """Film sans donnée collaborative : le poids réel de `collab` passe aux signaux de contenu."""
    w = dict(weights)
    share = w["collab"] / len(cfg.cold_start_targets)
    w["collab"] = 0.0
    for s in cfg.cold_start_targets:
        w[s] += share
    return w


def query_weights(
    art: Artifacts,
    idx: int,
    cfg: WeightConfig = DEFAULT_CONFIG,
    override: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Poids effectifs pour un film requête (règles ou poids imposés, puis cold start)."""
    w = {s: float(override.get(s, 0.0)) for s in SIGNALS} if override else smart_weights(art.catalog.iloc[idx], cfg)
    if not art.has_collab(idx):
        w = redistribute_cold_start(w, cfg)
    return w


def contributions(art: Artifacts, idx: int, weights: Mapping[str, float]) -> NDArray[np.float32]:
    """Contributions normalisées `w_k · M_k[idx, :] / Σ w` (8 × n). Leur somme = score."""
    w = np.array([weights[s] for s in SIGNALS], dtype=np.float32)
    total = float(w.sum())
    out = np.zeros((len(SIGNALS), art.n), dtype=np.float32)
    if total <= 0:
        return out
    for k, s in enumerate(SIGNALS):
        if w[k] > 0:
            out[k] = (w[k] / total) * art.similarity(s, idx)
    return out


def scores(
    art: Artifacts,
    idx: int,
    cfg: WeightConfig = DEFAULT_CONFIG,
    override: Mapping[str, float] | None = None,
) -> NDArray[np.float32]:
    """Score hybride de tous les films pour un film requête (chemin rapide, évaluation)."""
    out: NDArray[np.float32] = contributions(art, idx, query_weights(art, idx, cfg, override)).sum(axis=0)
    return out


def mmr(
    candidates: NDArray[np.intp],
    relevance: NDArray[np.float32],
    embeddings: NDArray[np.float32],
    k: int,
    diversity: float,
) -> list[int]:
    """Maximal Marginal Relevance : λ·pertinence − (1−λ)·max similarité aux films déjà choisis.

    `diversity` = 1 − λ (0 : classement d'origine, 1 : diversité maximale).
    """
    lam = 1.0 - diversity
    chosen: list[int] = []
    remaining = list(candidates)
    max_sim = np.full(len(embeddings), -np.inf, dtype=np.float32)
    while remaining and len(chosen) < k:
        rem = np.array(remaining)
        redundancy = np.where(np.isfinite(max_sim[rem]), max_sim[rem], 0.0)
        best = int(rem[np.argmax(lam * relevance[rem] - (1 - lam) * redundancy)])
        chosen.append(best)
        remaining.remove(best)
        max_sim = np.maximum(max_sim, embeddings @ embeddings[best])
    return chosen


def recommend(
    art: Artifacts,
    query: int | Sequence[int],
    k: int = 4,
    cfg: WeightConfig = DEFAULT_CONFIG,
    weights: Mapping[str, float] | None = None,
    diversity: float = 0.0,
    mask: NDArray[np.bool_] | None = None,
) -> RecommendationResult:
    """Les `k` films les plus proches d'un film (ou de plusieurs films) requête(s).

    `query` : position(s) dans `art.catalog` (pas des titres : ils ne sont pas uniques).
    Plusieurs films requêtes : moyenne de leurs scores (« j'ai aimé A, B et C »).
    `weights` : poids imposés (mode avancé), à la place des règles adaptatives.
    `diversity` ∈ [0, 1] : re-classement MMR des meilleurs candidats.
    `mask` : films autorisés (filtres décennie, genre…), appliqué après le score.
    """
    queries = [query] if isinstance(query, (int, np.integer)) else list(query)
    if not queries:
        raise ValueError("au moins un film requête est nécessaire")

    per_query_w = [query_weights(art, int(q), cfg, weights) for q in queries]
    contrib = np.mean([contributions(art, int(q), w) for q, w in zip(queries, per_query_w, strict=True)], axis=0)
    total = contrib.sum(axis=0)

    allowed = np.ones(art.n, dtype=bool) if mask is None else mask.copy()
    allowed[queries] = False
    ranked_scores = np.where(allowed, total, -np.inf)
    n_allowed = int(allowed.sum())

    pool = min(max(k, MMR_POOL) if diversity > 0 else k, n_allowed)
    if pool <= 0:
        top: list[int] = []
    else:
        cand = np.argpartition(ranked_scores, -pool)[-pool:]
        cand = cand[np.argsort(ranked_scores[cand])[::-1]]
        top = mmr(cand, total, art.embeddings["sbert"], k, diversity) if diversity > 0 else [int(i) for i in cand[:k]]

    cat = art.catalog
    items = []
    for i in top:
        row = cat.iloc[i]
        year = row["year"]
        items.append(
            Recommendation(
                idx=i,
                id=int(row["id"]),
                title=str(row["title"]),
                year=int(year) if pd.notna(year) and year > 0 else None,
                director=str(row["director"] or ""),
                score=float(total[i]),
                contributions={s: float(contrib[k_, i]) for k_, s in enumerate(SIGNALS)},
            )
        )

    mean_w = {s: float(np.mean([w[s] for w in per_query_w])) for s in SIGNALS}
    rules = sorted({r for q in queries for r in active_rules(cat.iloc[int(q)], cfg)}) if weights is None else []
    cold = any(not art.has_collab(int(q)) for q in queries)
    return RecommendationResult(items, mean_w, rules, cold)
